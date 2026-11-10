"""VoiceTrail event-conditioned trail selection and evidence rendering.

Choose the event-linked trajectory or the controlled emotion-label baseline
using the supplied source-grounded memory observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping,Sequence
from .trajectory_selector import select as select_trajectory
POLICY="VOXATRAIL_TRAIL_BUNDLE_V7_20261029"
@dataclass(frozen=True)
class TrailChoice:
    enabled:bool
    observations:tuple[dict,...]
    reasons:dict
    def receipt(self):
        return dict(policy=POLICY,enabled=self.enabled,
            observation_ids=[o["obs_id"] for o in self.observations],reasons=self.reasons)
def choose(question:str,eligible:Sequence[Mapping],full_scores:Mapping[str,float],*,
           enabled:bool=True,flat_scores:Mapping[str,float]|None=None)->TrailChoice:
    rows=[dict(o) for o in eligible]
    ids={o["obs_id"] for o in rows}
    if len(ids)!=len(rows) or set(full_scores)!=ids:raise ValueError("EMOTION_SCORE_COVERAGE")
    if not rows:return TrailChoice(enabled,(),{})
    if enabled:
        result=select_trajectory(question,rows,dict(full_scores))
        return TrailChoice(True,tuple(result["observations"]),dict(result["reasons"]))
    if flat_scores is None or set(flat_scores)!=ids:
        raise ValueError("TRAIL_OFF_REQUIRES_INDEPENDENT_LABEL_ONLY_E5_SCORES")
    ranked=sorted(rows,key=lambda o:(-flat_scores[o["obs_id"]],o["obs_id"]))[:15]
    return TrailChoice(False,tuple(ranked),{o["obs_id"]:"FLAT_EMOTION_LABEL_E5" for o in ranked})
def flat_labels(obs:Sequence[Mapping])->str:
    labels=[]
    for o in obs:
        value=o.get("emotion",[])
        if isinstance(value,str):value=[value]
        if not isinstance(value,(list,tuple)) or not all(isinstance(v,str) for v in value):
            raise ValueError("INVALID_EMOTION_LABEL")
        labels.extend(x.strip() for x in value if x.strip())
    return "\n".join("- "+s for s in sorted(labels,key=str.casefold))
