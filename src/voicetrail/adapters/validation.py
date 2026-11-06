"""Source turn identity and first-person emotion evidence verification."""
import re

def resolve_writer_turn_id(raw_id: object, session_id: str, sources: dict) -> str | None:
    """Normalize valid turn IDs within the current session without inventing sources.

    Writer runs on one Session at a time. The sole repair permitted is a
    four-digit turn suffix inside THAT session, and only when the turn exists.
    This does not change who was speaking or what they actually said.
    """
    if isinstance(raw_id, str) and raw_id in sources:
        return raw_id
    if not isinstance(raw_id, str):
        return None
    match = re.fullmatch(r"(?:session_|turn_)(\d{4})", raw_id.strip())
    if not match:
        return None
    candidate = session_id + ":" + match.group(1)
    return candidate if candidate in sources else None


def verify_firsthand_emotion_quote(quote: str, source_text: str) -> bool:
    """Require a literally quoted, explicit first-person seeker state.

    Prevent third-person descriptions in a seeker utterance (e.g. 'he froze')
    and quoted speech attributed to someone else from becoming seeker emotion.
    This is intentionally conservative and may discard implicit emotions.
    """
    q = quote.casefold().strip()
    s = source_text.casefold()
    if not q or q not in s:
        return False
    # A direct quote preceded by a quotation mark belongs to quoted speech,
    # not automatically to the seeker who reports it.
    offset = s.find(q)
    if q.startswith(("i ", "i'", "i’m")) and s[:offset].rstrip().endswith(('"', "“", "'")):
        return False
    patterns = (
        r"\b(?:i feel|i felt|i hate|i worry|i guess|i wish)\b",
        r"\bi(?:'m| am| was) (?:worried|anxious|afraid|scared|angry|tired|sick|upset|nervous|frustrated|exhausted|overwhelmed)\b",
        r"\bmy hands\b",
        r"\b(?:makes me feel|made me feel|it feels like)\b",
        r"\bif i get\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)
