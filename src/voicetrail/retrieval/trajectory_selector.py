"""VoiceTrail event-conditioned emotion trajectory selection.

Score eligible source-grounded observations over the full historical pool,
then prioritize temporal anchors, event-linked sessions, and relevance within
the 15-observation budget. Exact stored event IDs define event membership.
"""
from __future__ import annotations
from collections import defaultdict
import re

K = 15
DATE_PER_DAY = 2
SEED_COUNT = 3
EVENT_EXPANSION_MAX = 4
ASPECT_BRIDGE_MAX = 3
ASPECT_MIN_SIM = 0.88
ASPECT_MIN_QUERY_GAP = 0.075
DATES = re.compile(r"(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)")

def select(question: str, eligible: list[dict], scores: dict[str, float],
           vectors: dict[str, object] | None = None, *, limit: int = K) -> dict:
    """Pure deterministic selection from supplied question/source-only model scores."""
    if limit != K:
        raise ValueError("UNFROZEN_ET_SLOT_COUNT")
    if not eligible:
        raise ValueError("NO_LEGAL_EMOTIONS")
    ids={o["obs_id"] for o in eligible}
    if len(ids)!=len(eligible) or set(scores)!=ids:
        raise ValueError("ET_SCORE_COVERAGE_OR_DUPLICATE")
    ranked=sorted(eligible,key=lambda o:(-scores[o["obs_id"]],o["obs_id"]))
    picked=[]; chosen=set(); reasons={}
    def add(o, why):
        oid=o["obs_id"]
        if oid not in chosen and len(picked)<K:
            picked.append(o);chosen.add(oid);reasons[oid]=why
            return True
        return False

    # 1. Explicit date-scoped anchors, before any TopK truncation.
    # Prefer separately sourced turns and don't require one common Event ID.
    date_anchors=[d for d in dict.fromkeys(DATES.findall(question))]
    date_anchors=date_anchors[:K//DATE_PER_DAY]
    for date in date_anchors:
        days=[o for o in ranked if o["record_time"]==date]
        selected_turns=set()
        for _ in range(DATE_PER_DAY):
            possible=[o for o in days if o["obs_id"] not in chosen]
            if not possible: break
            possible.sort(key=lambda o:(-scores[o["obs_id"]]+(0.025 if o["source_turn_id"] in selected_turns else 0),o["obs_id"]))
            winner=possible[0]
            if add(winner,"EXPLICIT_DATE_ANCHOR:"+date):
                selected_turns.add(winner["source_turn_id"])

    # 2. Retain the top query matches as direct anchors.
    for o in ranked[:SEED_COUNT]:
        add(o,"QUERY_E5_SEED")

    # 3. Expand selected Event groups from the full history, before K.
    # Expand across sessions using the event identities stored by the Writer.
    groups=defaultdict(list)
    for o in eligible:
        if o.get("event_id") and o.get("emotion_assertion")=="EXPERIENCED":
            groups[o["event_id"]].append(o)
    base_eventids=list(dict.fromkeys(o["event_id"] for o in picked if o.get("event_id")))
    base_eventids.sort(key=lambda eid:(-max(scores[o["obs_id"]] for o in picked if o.get("event_id")==eid),eid))
    added=0
    for eid in base_eventids:
        if added>=EVENT_EXPANSION_MAX:break
        members=groups[eid]
        if len({o["session_id"] for o in members})<2:continue
        present={o["session_id"] for o in picked if o.get("event_id")==eid}
        missing_by_session=defaultdict(list)
        for o in members:
            if o["session_id"] not in present and o["obs_id"] not in chosen:
                missing_by_session[o["session_id"]].append(o)
        # One evidence observation per previously missing Session, prioritize
        # relevant semantic match; keep earliest+latest for long coverage.
        candidates=[max(rows,key=lambda o:(scores[o["obs_id"]],o["obs_id"]))
                    for rows in missing_by_session.values()]
        candidates.sort(key=lambda o:(o["record_time"],o["obs_id"]))
        ordering=[]
        if candidates:ordering.append(candidates[0])
        if len(candidates)>1:ordering.append(candidates[-1])
        ordering += sorted([o for o in candidates if o not in ordering],
                            key=lambda o:(-scores[o["obs_id"]],o["obs_id"]))
        for o in ordering:
            if added>=EVENT_EXPANSION_MAX or len(picked)>=K:break
            if add(o,"SAME_EVENT_MISSING_SESSION:"+eid):added+=1

    # 4. Cross-Event, same-aspect CANDIDATES are not merged. Embedding
    # similarity to question anchors is supplementary; it is NOT identity.
    if vectors:
        import numpy as np
        top=ranked[:SEED_COUNT]
        anchor_vec=[vectors[o["obs_id"]] for o in top if o["obs_id"] in vectors]
        anchor_events={o.get("event_id") for o in top}
        best=scores[ranked[0]["obs_id"]]
        candidates=[]
        for o in ranked:
            if o["obs_id"] in chosen or scores[o["obs_id"]]<best-ASPECT_MIN_QUERY_GAP or o["obs_id"] not in vectors:
                continue
            if o.get("event_id") in anchor_events and o.get("event_id") is not None:
                continue
            similarity=max((float(np.dot(vectors[o["obs_id"]],v)) for v in anchor_vec),default=0.0)
            if similarity>=ASPECT_MIN_SIM:
                candidates.append((scores[o["obs_id"]],similarity,o))
        candidates.sort(key=lambda x:(-x[0],-x[1],x[2]["obs_id"]))
        siblings=0
        for _,sim,o in candidates:
            if siblings>=ASPECT_BRIDGE_MAX or len(picked)>=K:break
            if add(o,"RELATED_ASPECT_DIFFERENT_EVENT_UNCONFIRMED"):
                siblings+=1

    # 5. Fill remaining budget in original E5 relevance order.
    for o in ranked:
        if len(picked)>=K:break
        add(o,"GLOBAL_E5_BACKFILL")
    if len(picked)!=min(K,len(eligible)):
        raise AssertionError("ET_RETRIEVAL_WRONG_COUNT")
    return dict(observations=picked,ids=[o["obs_id"] for o in picked],reasons=reasons,
        explicit_dates=date_anchors,seed_k=SEED_COUNT,distinct_selected_dates=len({o["record_time"] for o in picked}),
        related_aspect_not_identity=True)
