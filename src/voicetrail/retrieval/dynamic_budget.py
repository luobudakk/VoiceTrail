"""VoxaTrail mainline query-time evidence budgeting (V6, source-only).

Default is dynamic after the 2026-10-29 DEV pilot. For legacy ablations
use mode=fixed_legacy. Only selects from eligible ranked candidates;
never infers event identity and never accesses benchmark type/Gold.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import json

POLICY_VERSION = "VOXATRAIL_MAIN_DYNAMIC_V6_20261029"
DEFAULT_MODE = "dynamic"
MODES = ("dynamic", "fixed_legacy")
BUDGETS = {
    "DUAL_EVENT": (18, 10, 15),
    "DATE_ORDINAL": (16, 10, 15),
    "DAY_INTERVAL": (16, 10, 15),
    "MULTI_STAGE": (16, 10, 15),
    "EMOTION_TRANSITION": (12, 8, 12),
    "FUTURE_UNSEEN": (8, 6, 6),
    "SINGLE_EMOTION": (8, 6, 6),
    "GENERAL": (12, 8, 12),
}
LEGACY = (20, 12, 15)

def route_question(question: str) -> str:
    q = question.casefold()
    if "next future follow-up" in q or "abstain if" in q or "not yet observed" in q:
        return "FUTURE_UNSEEN"
    if "two distinct event histories" in q or ("event a" in q and "event b" in q):
        return "DUAL_EVENT"
    if "fourth follow-up" in q or ("record date of" in q and "follow-up" in q):
        return "DATE_ORDINAL"
    if "how many days" in q or ("days" in q and "follow-up" in q):
        return "DAY_INTERVAL"
    if "first" in q and "middle" in q and "last" in q:
        return "MULTI_STAGE"
    if ("earlier development" in q or "later development" in q
            or ("first conversation" in q and "latest" in q)
            or ("earliest" in q and "latest" in q)
            or "between the conversations" in q or "earlier then later" in q):
        return "EMOTION_TRANSITION"
    if "what emotion" in q or "what feeling" in q:
        return "SINGLE_EMOTION"
    return "GENERAL"

def budget_for_question(question: str, *, mode: str=DEFAULT_MODE):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QUESTION_REQUIRED")
    if mode not in MODES:
        raise ValueError("UNKNOWN_BUDGET_MODE")
    kind = route_question(question)
    return kind, (BUDGETS[kind] if mode=="dynamic" else LEGACY)

@dataclass(frozen=True)
class EvidenceSelection:
    policy_version: str
    mode: str
    route: str
    budget: tuple[int,int,int]
    raw_turns: tuple[dict[str,Any],...]
    facts: tuple[dict[str,Any],...]
    observations: tuple[dict[str,Any],...]
    dropped_fact_source_turns: tuple[str,...]
    candidate_count: tuple[int,int,int]
    def receipt(self):
        return dict(
            policy_version=self.policy_version, mode=self.mode, route=self.route,
            budget=list(self.budget),candidate_count=list(self.candidate_count),
            selected_count=[len(self.raw_turns),len(self.facts),len(self.observations)],
            raw_ids=[t["turn_id"] for t in self.raw_turns],
            fact_ids=[f["fact_id"] for f in self.facts],
            observation_ids=[o["obs_id"] for o in self.observations],
            dropped_fact_source_turns=list(self.dropped_fact_source_turns))

def select_evidence(question: str, raw20: Sequence[Mapping[str,Any]],
       fact12: Sequence[Mapping[str,Any]], emotion15: Sequence[Mapping[str,Any]],
       *, cutoff_record_date: str, mode: str=DEFAULT_MODE) -> EvidenceSelection:
    """Ranked inputs, fail closed on future evidence or duplicate IDs.

    Mainline dynamic: same budget/pruning as frozen V6 DEV pilot.
    Legacy mode retains raw20/fact12/emotion15 with no fact source dedup.
    """
    if not cutoff_record_date:
        raise ValueError("CUTOFF_REQUIRED")
    route,budget=budget_for_question(question,mode=mode)
    if len(raw20)>20 or len(fact12)>12 or len(emotion15)>15:
        raise ValueError("EXCESS_CANDIDATES")
    def check(items,key):
        data=[dict(x) for x in items]
        ids=[x[key] for x in data]
        if len(ids)!=len(set(ids)):
            raise ValueError("DUPLICATE_SOURCE_ID")
        if any(x["record_time"]>cutoff_record_date for x in data):
            raise ValueError("FUTURE_EVIDENCE")
        return data
    raw=check(raw20,"turn_id")
    facts=check(fact12,"fact_id")
    emotions=check(emotion15,"obs_id")
    keep_raw=raw[:budget[0]]
    keep_obs=emotions[:budget[2]]
    removed=[]
    if mode=="fixed_legacy":
        keep_facts=facts[:budget[1]]
    else:
        r_sources={x.get("source_turn_id",x["turn_id"]) for x in keep_raw}
        keep_facts=[]
        for f in facts:
            if f["source_turn_id"] in r_sources:
                removed.append(f["source_turn_id"])
                continue
            keep_facts.append(f)
            if len(keep_facts)==budget[1]:break
    return EvidenceSelection(
        POLICY_VERSION,mode,route,budget,tuple(keep_raw),tuple(keep_facts),
        tuple(keep_obs),tuple(removed),(len(raw),len(facts),len(emotions)))

def short_quote(q: str, n: int=130) -> str:
    q=" ".join((q or "").split())
    if len(q)<=n:return q
    i=q[:n].rfind(" ")
    return q[:i if i>=int(n*.65) else n]

def render_compact_observations(observations: Sequence[Mapping[str,Any]]) -> str:
    out=[]
    for o in sorted(observations, key=lambda x:(x["record_time"],x["session_id"],x["obs_id"])):
        if o.get("event_id") and o.get("link_status")!="LINKED_MODEL_CLAIM":
            raise ValueError("EVENT_NOT_MODEL_LINKED")
        if o.get("time_class") not in ("CURRENT","PAST_RECALL","FUTURE_ANTICIPATED","UNKNOWN"):
            raise ValueError("UNKNOWN_TIME_CLASS")
        row=dict(id=o["obs_id"],t=o["record_time"],e=", ".join(o["emotion"]),
            turn=o["source_turn_id"],q=short_quote(o.get("emotion_quote") or ""),
            ev=o.get("event_id"),tc=o["time_class"],session=o["session_id"])
        if o.get("emotion_assertion")!="EXPERIENCED":
            row["assertion"]=o.get("emotion_assertion")
        for field in ("aspect","target","event_anchor"):
            if o.get(field):
                row[field]=o[field]
        out.append(json.dumps(row,ensure_ascii=False,separators=(",",":")))
    return "\n".join(out)

def assemble_reader_context(question: str,
        raw20: Sequence[Mapping[str,Any]], fact12: Sequence[Mapping[str,Any]],
        emotion15: Sequence[Mapping[str,Any]], *, cutoff_record_date: str,
        mode: str=DEFAULT_MODE) -> tuple[str,dict[str,Any]]:
    """VoxaTrail mainline Reader input assembly. Defaults to dynamic policy.

    Legacy frozen research scripts deliberately do NOT call this method.
    Caller owns generation; function returns ready-to-embed evidence and IDs.
    """
    selected=select_evidence(question,raw20,fact12,emotion15,
        cutoff_record_date=cutoff_record_date,mode=mode)
    return render_evidence_context(selected),selected.receipt()

def render_evidence_context(selection: EvidenceSelection) -> str:
    """Direct Reader evidence block using the selected current main policy."""
    raw="\n".join(
        f'[{x["turn_id"]}][date={x["record_time"]}][role={x["speaker"]}] {x["text"]}'
        for x in selection.raw_turns)
    facts=[dict(text=x["text"],date=x["record_time"],turn=x["source_turn_id"])
           for x in selection.facts]
    return ("ORIGINAL SOURCE TURNS (Raw%d):\n" % len(selection.raw_turns)
        +raw+"\nFACTS:\n"+json.dumps(facts,ensure_ascii=False,separators=(",",":"))
        +"\nEMOTION OBSERVATIONS:\n"+render_compact_observations(selection.observations))
