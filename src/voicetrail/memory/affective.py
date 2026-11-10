"""Conservative assertion and temporal interpretation of quoted feelings."""
from __future__ import annotations

import re

ASSERTIONS = frozenset({"EXPERIENCED", "HYPOTHETICAL", "INTENDED", "UNCLEAR"})
TIME_CLASSES = frozenset({"CURRENT", "PAST_RECALL", "FUTURE_ANTICIPATED", "UNKNOWN"})

_HYPOTHETICAL = re.compile(
    r"\b(?:if i\b|what if\b|i (?:might|would|could|may) feel\b|"
    r"i (?:might|would|could) be\b|imagine (?:i|myself)\b)", re.I)
_INTENDED = re.compile(
    r"\bi (?:will|shall|expect to|plan to|am going to|'m going to) "
    r"(?:feel|be)\b", re.I)
_PAST = re.compile(
    r"\b(?:i (?:felt|was|used to feel|remember feeling)\b|"
    r"back then\b|last (?:month|year|week|night)\b|yesterday\b)", re.I)
_FIRSTHAND = re.compile(
    r"\b(?:i (?:feel|felt|am|was)\b|i['\u2019]m\b|"
    r"makes? me feel\b|made me feel\b|my hands\b)", re.I)


def classify_affective_quote(quote: str, assertion: object = None,
                             time_class: object = None) -> tuple[str, str]:
    """Combine Writer labels with explicit lexical evidence, favoring safety.

    An experienced emotion must have a direct first-person expression; future
    or conditional claims are never promoted to EXPERIENCED by Writer output.
    """
    desired_assertion = assertion if assertion in ASSERTIONS else "UNCLEAR"
    desired_time = time_class if time_class in TIME_CLASSES else "UNKNOWN"
    if _HYPOTHETICAL.search(quote):
        return "HYPOTHETICAL", "UNKNOWN"
    if _INTENDED.search(quote):
        return "INTENDED", "FUTURE_ANTICIPATED"
    if not _FIRSTHAND.search(quote):
        return "UNCLEAR", desired_time
    if desired_assertion in {"HYPOTHETICAL", "INTENDED", "UNCLEAR"} and assertion in ASSERTIONS:
        final_assertion = desired_assertion
    else:
        final_assertion = "EXPERIENCED"
    if _PAST.search(quote):
        final_time = "PAST_RECALL"
    elif desired_time == "PAST_RECALL":
        final_time = "PAST_RECALL"
    else:
        final_time = "CURRENT"
    return final_assertion, final_time
