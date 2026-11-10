"""Conservative, source-attested event identity for longitudinal emotion records.

Only an explicit stable event_key with an exact source anchor can become a
cross-session identifier. Neither embedding similarity nor a generic event
label is used to merge different experiences.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_STOP = frozenset({
    "the", "a", "an", "my", "our", "this", "that", "about", "event", "feeling",
    "emotion", "situation", "experience", "issue", "thing", "work", "life",
})
_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
_HAN = re.compile(r"^[\u3400-\u9fff]+$")
_HAN_SPACES = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")

def _tokens(value: str) -> list[str]:
    normalized = _HAN_SPACES.sub("", unicodedata.normalize("NFKC", value).casefold())
    pieces = _TOKEN.findall(normalized)
    tokens = []
    for item in pieces:
        if _HAN.fullmatch(item) and len(item) >= 4:
            tokens.extend(item[i:i + 2] for i in range(len(item) - 1))
        else:
            tokens.append(item)
    return tokens


def stable_event_id(user_id: str, event_key: object, event_anchor: object,
                    source_transcripts: list[str]) -> str | None:
    """Return a user-scoped exact-match ID when a distinctive key is attested.

    The anchor must be a literal phrase from a seeker utterance in the current
    source turn; at least three distinctive key tokens must appear in it.
    Different keys NEVER become equivalent by semantic similarity. Writers must
    reuse the same canonical key for the same continuing event.
    """
    if not isinstance(event_key, str) or not isinstance(event_anchor, str):
        return None
    tokens = _tokens(event_key)
    distinct = {w for w in tokens if (len(w) > 2 or
                (len(w) == 2 and _HAN.fullmatch(w)))
                and w not in _STOP and not w.isdigit()}
    anchor = unicodedata.normalize("NFKC", event_anchor).casefold().strip()
    if len(distinct) < 3 or len(anchor) < 6:
        return None
    if not any(anchor in unicodedata.normalize("NFKC", text).casefold()
               for text in source_transcripts):
        return None
    if len(distinct.intersection(_tokens(anchor))) < 3:
        return None
    normalized = "-".join(tokens)
    digest = hashlib.sha256((user_id + "\0" + normalized).encode("utf-8")).hexdigest()[:20]
    return "evt:" + digest
