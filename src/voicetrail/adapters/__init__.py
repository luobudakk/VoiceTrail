'''Stage adapters for voice input, memory extraction and evidence-grounded QA.'''
from .live import LiveStages
from .validation import resolve_writer_turn_id, verify_firsthand_emotion_quote
__all__ = ['LiveStages', 'resolve_writer_turn_id', 'verify_firsthand_emotion_quote']
