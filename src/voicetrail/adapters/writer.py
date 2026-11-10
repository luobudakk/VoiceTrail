"""Source-bound factual and affective memory extraction."""
import re
import json
from typing import Any
from ..scheduler import Session
from ..memory.event_identity import stable_event_id
from ..memory.affective import classify_affective_quote
from .validation import resolve_writer_turn_id, verify_firsthand_emotion_quote


def _source_span(value: object, source: str) -> str | None:
    """Persist aspect/target only when supported verbatim by the source turn."""
    if not isinstance(value, str):
        return None
    value = value.strip()[:160]
    return value if value and value.casefold() in source.casefold() else None


class WriterAdapter:
    def writer(self, session: Session, rows: tuple[dict, ...]) -> dict[str, Any]:
        body = "\n".join(
            "[" + session.session_id + ":" + str(r["turn_index"]).zfill(4)
            + "][" + r["role"] + "] " + r["transcript"] for r in rows
        )
        system = (
            "Extract source-grounded factual and affective memories from the "
            "provided transcript only. Return a JSON object with facts and emotions. "
            "Never attribute another speaker's emotion to the seeker. "
            "Distinguish actually experienced emotions, remembered feelings, "
            "hypotheticals and anticipated emotions. No invented event identities."
        )
        prompt = (
            "Return JSON only. Extract up to 7 facts with source_turn_id and text; "
            "and up to 6 seeker affective observations with source_turn_id, emotion "
            "(list), emotion_quote (exact nonempty verbatim substring of a seeker "
            "turn), emotion_assertion (EXPERIENCED, HYPOTHETICAL, INTENDED or "
            "UNCLEAR), time_class (CURRENT, PAST_RECALL, FUTURE_ANTICIPATED or "
            "UNKNOWN), optional aspect and target (verbatim substrings of the "
            "source turn), and event_label. For a genuinely identifiable, "
            "continuing event, also provide event_key (specific normalized durable "
            "event reference, exactly reusable across sessions) and event_anchor "
            "(verbatim phrase from a SEEKER utterance establishing that event). "
            "Omit event_key if the identity is vague or uncertain; never merge "
            "events only because they share a domain or emotion. "
            "Use only literal source_turn_id values from the transcript. "
            "Examples: 'I felt nervous last week' = EXPERIENCED, PAST_RECALL; "
            "'If I fail I would feel scared' = HYPOTHETICAL, UNKNOWN; "
            "'I will feel nervous' = INTENDED, FUTURE_ANTICIPATED.\n"
            "TRANSCRIPTS:\n" + body
        )
        response = self.requests.post(
            self.writer_url,
            headers=self.writer_headers,
            json={
                "model": self.writer_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2000,
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
                "text": r["transcript"], "turn_order": r["turn_index"],
            }
            for r in rows
        ]
        facts = []
        affect = []
        rejected = []
        source_id_repairs = []
        for item in data["facts"][:7]:
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
        for item in data["emotions"][:6]:
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
            assertion, time_class = classify_affective_quote(
                quote, item.get("emotion_assertion"), item.get("time_class"))
            event_key = item.get("event_key")
            event_anchor = item.get("event_anchor")
            src = sources[sid]["transcript"]
            global_event_id = stable_event_id(
                session.user_id, event_key, event_anchor, [src])
            label = re.sub(r"[^a-z0-9]+", "_",
                           str(item.get("event_label") or "").casefold()).strip("_")[:30]
            # Backward compatibility: unverified labels remain session-local.
            event_id = global_event_id or (session.session_id + ":" + label if label else None)
            affect.append({
                "obs_id": session.session_id + ":o" + str(len(affect) + 1).zfill(3),
                "seeker_id": session.user_id, "session_id": session.session_id,
                "record_time": session.date, "source_turn_id": sid,
                "emotion": [str(v) for v in labels[:4]], "emotion_quote": quote,
                "emotion_assertion": assertion, "time_class": time_class,
                "event_id": event_id,
                "event_key": event_key if global_event_id else None,
                "event_anchor": event_anchor if global_event_id else None,
                "aspect": _source_span(item.get("aspect"), src),
                "target": _source_span(item.get("target"), src),
                "link_status": "LINKED_MODEL_CLAIM" if event_id else None,
            })
        # Sessions without explicit emotions are valid memory sessions.
        # Source validation may legitimately reject every emotion candidate.
        return {
            "user_id": session.user_id, "session_id": session.session_id,
            "date": session.date, "raw": raw, "facts": facts, "emotions": affect,
            "rejected": rejected, "source_id_repairs": source_id_repairs,
            "writer_model": obj.get("model"),
            "writer_usage": obj.get("usage"),
            "writer_policy": "SOURCE_ATTESTED_LONGITUDINAL_WRITER_V2",
        }
