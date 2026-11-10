"""VoiceTrail deterministic schema- and entity-aware evidence ranking.

Source-derived life-domain and entity cues adjust E5 relevance scores with
a lightweight, rule-based routing policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence
import re

VERSION="VOXATRAIL_SCHEMA_ENTITY_SOURCE_ONLY_V7_20261029"
DOMAINS={
"work":("work","career","job","boss","office","thesis","research","project","interview","colleague","university","school","论文","工作"),
"finance":("money","salary","income","budget","bank","invest","loan","debt","财务","投资"),
"relationships":("family","friend","partner","relationship","sister","brother","mother","father","wife","husband","wedding","social","support group","朋友","家庭"),
"health":("health","hospital","doctor","medical","therapy","sleep","pain","depression","illness","exercise","anxiety","健康","睡眠"),
"goals":("goal","plan","future","hope","dream","aspiration","target","achievement","deadline","目标","计划"),
"daily_life":("daily","travel","camp","museum","paint","pottery","concert","cook","hobby","holiday","routine","爱好","日常"),
"knowledge":("learn","study","exam","course","book","knowledge","algorithm","skill","reading","考试","学习"),
}
IGNORE={"What","When","Where","Who","Why","How","Did","Does","Was","Were","The","This","That","These","Those","Which","After","Before","During","Since","User","Session","Event","Emotion","Unknown","First","Last"}
PROPER=re.compile(r"(?<!\w)(?:[A-Z][a-z]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]{2,}|[A-Z]{2,})){0,2}(?!\w)")
def contains(text:str,term:str)->bool:
    if any("\u3400"<=c<="\u9fff" for c in term):return term.casefold() in text.casefold()
    return re.search(r"(?<!\w)"+re.escape(term)+r"(?!\w)",text,re.I) is not None
@lru_cache(maxsize=100000)
def slots(text:str)->tuple[str,...]:
    weights=[(k,sum(contains(text,word) for word in v)) for k,v in DOMAINS.items()]
    return tuple(k for k,n in sorted((x for x in weights if x[1]),key=lambda x:(-x[1],x[0]))[:2])
@lru_cache(maxsize=100000)
def names(text:str)->tuple[str,...]:
    out={}
    for m in PROPER.finditer(text):
        v=m.group();parts=v.split()
        if all(x in IGNORE for x in parts):continue
        out.setdefault(v.casefold(),v)
    return tuple(out.values())
def source_text(r:Mapping)->str:
    pieces=[r.get(k,"") for k in ("text","emotion_quote","aspect")]
    emo=r.get("emotion",[])
    if isinstance(emo,str):emo=[emo]
    return " ".join([x for x in pieces if isinstance(x,str)]+[x for x in emo if isinstance(x,str)])
@dataclass(frozen=True)
class SchemaRoute:
    enabled:bool
    query_slots:tuple[str,...]
    query_entities:tuple[str,...]
    policy:str=VERSION
    def delta(self,row:Mapping)->float:
        if not self.enabled:return 0.0
        text=source_text(row)
        return (0.08 if set(slots(text))&set(self.query_slots) else 0.0)+0.12*min(2,sum(contains(text,e) for e in self.query_entities))
    def receipt(self):return dict(policy=self.policy,enabled=self.enabled,slots=list(self.query_slots),entities=list(self.query_entities))
def route(question:str,legal_corpus:Sequence[Mapping],*,enabled:bool=True)->SchemaRoute:
    if not question.strip():raise ValueError("QUESTION_REQUIRED")
    if not enabled:return SchemaRoute(False,(),())
    catalog={v.casefold():v for row in legal_corpus for v in names(source_text(row))}
    entities=[]
    for n in sorted(names(question),key=lambda v:(-len(v),v)):
        v=catalog.get(n.casefold())
        if v and v not in entities:entities.append(v)
    return SchemaRoute(True,slots(question),tuple(entities[:8]))
def rerank(rows:Sequence[Mapping],scores:Mapping[str,float],id_key:str,decision:SchemaRoute)->list[dict]:
    ids=[r[id_key] for r in rows]
    if len(set(ids))!=len(ids) or set(scores)!=set(ids):raise ValueError("SCORES_MUST_COVER_LEGAL_POOL")
    return sorted((dict(r) for r in rows),key=lambda r:(-float(scores[r[id_key]])-decision.delta(r),str(r[id_key])))
def rescore(rows:Sequence[Mapping],scores:Mapping[str,float],id_key:str,decision:SchemaRoute)->dict[str,float]:
    ids=[r[id_key] for r in rows]
    if len(set(ids))!=len(ids) or set(scores)!=set(ids):raise ValueError("SCORES_MUST_COVER_LEGAL_POOL")
    return {r[id_key]:float(scores[r[id_key]])+decision.delta(r) for r in rows}
