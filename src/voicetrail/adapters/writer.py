"""Source-bound factual and affective memory extraction."""
import re
import json
from typing import Any
from ..scheduler import Session
from .validation import resolve_writer_turn_id, verify_firsthand_emotion_quote

class WriterAdapter:
    def writer(self, session: Session, rows: tuple[dict, ...]) -> dict[str, Any]:
        body = "\n".join(
            "[" + session.session_id + ":" + str(r["turn_index"]).zfill(4)
            + "][" + r["role"] + "] " + r["transcript"] for r in rows
        )
        system = (
            "You extract factual and affective memory exclusively from the supplied "
            "conversation transcripts. Output one JSON object with keys facts and emotions. "
            "Never invent facts or claim third-party or supporter emotions belong to the seeker. "
            "No markdown, explanations, or reasoning."
        )
        prompt = (
            "Extract up to 7 facts as objects with source_turn_id and text; "
            "and up to 6 firsthand seeker emotion observations as objects with "
            "source_turn_id, emotion (list of labels), emotion_quote (an EXACT "
            "nonempty verbatim substring of the stated seeker turn) and event_label. "
            "Only use literal source_turn_id values from the transcript. "
            "Never reuse speaker or event identities without direct source evidence. "
            "Response JSON only.\nTRANSCRIPTS:\n" + body
        )
        response = self.requests.post(
            self.writer_url,
            headers=self.writer_headers,
            json={
                "model": self.writer_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1400,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        obj = response.json()
        text = str(obj["choices"][0]["message"]["content"]).strip()
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        if not isinstance(data.get("facts"), list) or not isinstance(data.get("emotions"), list):
            raise ValueError("unusable Writer output")
        sources = {
            session.session_id + ":" + str(r["turn_index"]).zfill(4): r for r in rows
        }
        raw = [
            {
                "turn_id": session.session_id + ":" + str(r["turn_index"]).zfill(4),
                "seeker_id": session.user_id, "record_time": session.date,
                "session_id": session.session_id, "speaker": r["role"],
                "text": r["transcript"],
            }
            for r in rows
        ]
        facts = []
        affect = []
        rejected = []
        source_id_repairs = []
        for item in data["facts"]:
            original_sid = item.get("source_turn_id")
            sid = resolve_writer_turn_id(original_sid, session.session_id, sources)
            if sid and sid != original_sid:
                source_id_repairs.append({"before": original_sid, "after": sid, "kind": "fact"})
            value = str(item.get("text") or "").strip()
            if sid not in sources or not value:
                rejected.append({"type": "fact_bad_source", "source": sid})
                continue
            facts.append({
                "fact_id": session.session_id + ":f" + str(len(facts) + 1).zfill(3),
                "seeker_id": session.user_id, "session_id": session.session_id,
                "record_time": session.date, "source_turn_id": sid, "text": value[:800],
            })
        for item in data["emotions"]:
            original_sid = item.get("source_turn_id")
            sid = resolve_writer_turn_id(original_sid, session.session_id, sources)
            if sid and sid != original_sid:
                source_id_repairs.append({"before": original_sid, "after": sid, "kind": "emotion"})
            quote = str(item.get("emotion_quote") or "").strip()
            labels = item.get("emotion")
            if isinstance(labels, str):
                labels = [labels]
            if (sid not in sources or sources[sid]["role"] != "seeker" or not quote
                    or not verify_firsthand_emotion_quote(quote, sources[sid]["transcript"])
                    or not isinstance(labels, list) or not labels):
                rejected.append({"type": "emotion_bad_source_role_or_quote", "source": sid})
                continue
            label = re.sub(r"[^a-z0-9]+", "_",
                           str(item.get("event_label") or "").casefold()).strip("_")[:30]
            event_id = session.session_id + ":" + label if label else None
            affect.append({
                "obs_id": session.session_id + ":o" + str(len(affect) + 1).zfill(3),
                "seeker_id": session.user_id, "session_id": session.session_id,
                "record_time": session.date, "source_turn_id": sid,
                "emotion": [str(v) for v in labels[:4]], "emotion_quote": quote,
                "emotion_assertion": "EXPERIENCED", "time_class": "CURRENT",
                "event_id": event_id,
                "link_status": "LINKED_MODEL_CLAIM" if event_id else None,
            })
        if not facts or not affect:
            raise ValueError("Writer produced no validated facts or emotions")
        return {
            "user_id": session.user_id, "session_id": session.session_id,
            "date": session.date, "raw": raw, "facts": facts, "emotions": affect,
            "rejected": rejected, "source_id_repairs": source_id_repairs,
            "writer_model": obj.get("model"),
            "writer_usage": obj.get("usage"),
            "writer_policy": "SOURCE_VALIDATED_SESSION_WRITER_V1",
        }
