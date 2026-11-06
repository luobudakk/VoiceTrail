"""VoxaTrail V7 research-ready query pipeline.

Full legal bank → source-only Schema/Entity router → E5 retrieval/Neighbor
→ single Trail bundle → frozen V6 dynamic budgets & rendering.
No Writer, Reader, Gold, E5 encoder, external service or network I/O.
All scores must come from the same caller-frozen E5 revision.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping,Sequence
from .schema_entity import route,rerank,rescore
from .trail_bundle import choose,flat_labels
from .dynamic_budget import (
    POLICY_VERSION as BUDGET_POLICY, select_evidence,render_evidence_context,
)
PIPELINE_POLICY="VOXATRAIL_SCHEMA_TRAIL_PIPELINE_V7_20261029"


@dataclass(frozen=True)
class QueryEvidence:
    context:str
    receipt:dict


def _legal(rows:Sequence[Mapping],key:str,cutoff:str,user_id:str)->list[dict]:
    out=[]
    ids=set()
    for item in rows:
        r=dict(item)
        rid=r[key]
        if rid in ids:raise ValueError("DUPLICATE_ID:"+key)
        ids.add(rid)
        if r.get("seeker_id",r.get("user_id"))!=user_id:
            raise ValueError("CROSS_USER_MEMORY")
        if not r.get("record_time") or r["record_time"]>cutoff:
            raise ValueError("FUTURE_OR_UNDATED_MEMORY")
        out.append(r)
    return out


def _raw_candidates(rows:list[dict],scores:Mapping[str,float],decision,*,
                    neighbor:bool=True)->tuple[list[dict],dict]:
    ranked=rerank(rows,scores,"turn_id",decision)
    if not neighbor:
        out=ranked[:20]
        return out,dict(policy="E5_ALL_ROLE_DIRECT_TOP20",seed_ids=[],neighbor_ids=[])
    seekers=[r for r in ranked if r.get("speaker")=="seeker"]
    # If a user has fewer than five legal Seeker turns, use the available ones.
    seeds=seekers[:5]
    ordered=sorted(rows,key=lambda r:(
        r["record_time"],r["session_id"],
        int(r.get("original_turn_idx",r.get("turn_order",0))),r["turn_id"]))
    bysession=defaultdict(list)
    for row in ordered:bysession[row["session_id"]].append(row)
    pos={r["turn_id"]:i for group in bysession.values() for i,r in enumerate(group)}
    chosen=[];seen=set();neighbors=[]
    def add(r):
        if r["turn_id"] not in seen and len(chosen)<20:
            seen.add(r["turn_id"]);chosen.append(r)
    for s in seeds:add(s)
    for s in seeds:
        group=bysession[s["session_id"]]
        k=pos[s["turn_id"]]
        for offset in (-1,1):
            j=k+offset
            if 0<=j<len(group):
                row=group[j]
                if row["turn_id"] not in seen:neighbors.append(row["turn_id"])
                add(row)
    for r in ranked:add(r)
    return chosen,dict(policy="E5_SEEKER_TOP5_SAME_SESSION_PM1_ALL_ROLE_E5_FILL20",
        seed_ids=[r["turn_id"] for r in seeds],neighbor_ids=neighbors)


def _hold(rows:list[dict],id_key:str,ids:list[str],limit:int)->list[dict]:
    bank={r[id_key]:r for r in rows}
    if len(ids)!=len(set(ids)) or len(ids)>limit or any(i not in bank for i in ids):
        raise ValueError("INVALID_HELD_BASELINE_IDS")
    return [bank[i] for i in ids]


def assemble_query(question:str,*,user_id:str,cutoff_record_date:str,
        raw_pool:Sequence[Mapping],fact_pool:Sequence[Mapping],
        emotion_pool:Sequence[Mapping],
        raw_scores:Mapping[str,float],fact_scores:Mapping[str,float],
        emotion_scores:Mapping[str,float],
        flat_emotion_scores:Mapping[str,float]|None=None,
        schema_enabled:bool=True,trail_enabled:bool=True,
        emotion_enabled:bool=True,neighbor_enabled:bool=True,
        fact_enabled:bool=True,mode:str="dynamic",
        hold_full:Mapping|None=None)->QueryEvidence:
    """Compute FULL or a single controlled ablation.

    hold_full: Full receipt when turning off Emotion/Neighbor/Fact, to pin
    all other final evidence IDs and avoid source-dedup side effects.
    For Schema ON/OFF do NOT pass hold_full: candidate ranking must be free
    to change, otherwise Schema has no measurable effect.
    """
    if not question.strip() or not cutoff_record_date or not user_id:
        raise ValueError("QUERY_USER_CUTOFF_REQUIRED")
    raw=_legal(raw_pool,"turn_id",cutoff_record_date,user_id)
    facts=_legal(fact_pool,"fact_id",cutoff_record_date,user_id)
    emotions=_legal(emotion_pool,"obs_id",cutoff_record_date,user_id)
    for scores,items,key in ((raw_scores,raw,"turn_id"),(fact_scores,facts,"fact_id"),
                             (emotion_scores,emotions,"obs_id")):
        if set(scores)!={r[key] for r in items}:raise ValueError("E5_SCORE_COVERAGE:"+key)
    decision=route(question,[*raw,*facts,*emotions],enabled=schema_enabled)
    raw_ranked,raw_receipt=_raw_candidates(raw,raw_scores,decision,neighbor=neighbor_enabled)
    fact_ranked=rerank(facts,fact_scores,"fact_id",decision)[:12]
    et_scores=rescore(emotions,emotion_scores,"obs_id",decision)
    affect=choose(question,emotions,et_scores,enabled=trail_enabled,
                  flat_scores=flat_emotion_scores if emotion_enabled else None) if emotion_enabled else None
    et=list(affect.observations) if affect else []
    if hold_full:
        b=hold_full["budget_receipt"]
        if not neighbor_enabled:
            fact_ranked=_hold(facts,"fact_id",b["fact_ids"],12)
            et=_hold(emotions,"obs_id",b["observation_ids"],15)
        elif not fact_enabled:
            raw_ranked=_hold(raw,"turn_id",b["raw_ids"],20)
            et=_hold(emotions,"obs_id",b["observation_ids"],15)
        elif not emotion_enabled:
            raw_ranked=_hold(raw,"turn_id",b["raw_ids"],20)
            fact_ranked=_hold(facts,"fact_id",b["fact_ids"],12)
        else:
            raise ValueError("HOLD_ONLY_FOR_SINGLE_CHANNEL_ABLATION")
    if not fact_enabled:fact_ranked=[]
    if not emotion_enabled:et=[]
    selected=select_evidence(question,raw_ranked,fact_ranked,et,
        cutoff_record_date=cutoff_record_date,mode=mode)
    if hold_full and (not fact_enabled or not neighbor_enabled or not emotion_enabled):
        # Existing Full can source-dedup Facts; their ID selection must be pinned
        # exactly even when the Raw channel differs.
        b=hold_full["budget_receipt"]
        if fact_enabled:
            keep=_hold(facts,"fact_id",b["fact_ids"],12)
        else:keep=[]
        if not neighbor_enabled:
            # Keep Raw from the new arm, other channels from Full.
            selected=selected.__class__(**{**selected.__dict__,"facts":tuple(keep),
                "observations":tuple(_hold(emotions,"obs_id",b["observation_ids"],15))})
        elif not fact_enabled:
            selected=selected.__class__(**{**selected.__dict__,
                "raw_turns":tuple(_hold(raw,"turn_id",b["raw_ids"],20)),
                "observations":tuple(_hold(emotions,"obs_id",b["observation_ids"],15))})
        else:
            selected=selected.__class__(**{**selected.__dict__,
                "raw_turns":tuple(_hold(raw,"turn_id",b["raw_ids"],20)),"facts":tuple(keep)})
    content=render_evidence_context(selected)
    if emotion_enabled and not trail_enabled:
        # The V6 renderer would leak Event/Date/Quote: replace the entire affect
        # section with FE's emotion-only list. Raw/Fact dates are common controls.
        head=content.split("\nEMOTION OBSERVATIONS:\n",1)[0]
        content=head+"\nEMOTION OBSERVATIONS:\n"+flat_labels(selected.observations)
    if not emotion_enabled:
        content=content.split("\nEMOTION OBSERVATIONS:\n",1)[0]
    receipt=dict(pipeline=PIPELINE_POLICY,schema=decision.receipt(),
        trail=affect.receipt() if affect else dict(enabled=False,observation_ids=[]),
        neighbor=raw_receipt,budget_policy=BUDGET_POLICY,
        budget_receipt=selected.receipt(),cutoff=cutoff_record_date,user_id=user_id,
        flags=dict(schema=schema_enabled,trail=trail_enabled,emotion=emotion_enabled,
            neighbor=neighbor_enabled,fact=fact_enabled))
    return QueryEvidence(content,receipt)
