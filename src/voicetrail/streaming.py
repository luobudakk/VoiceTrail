"""Four-stage incremental spoken-query preparation for VoiceTrail.

Applications provide partial ASR transcripts or separately transcribed WAV
chunks through listen_audio/speech_tail_audio. Anticipation prepares early
candidates; Searching uses the resolved utterance and time-eligible history.
The batch scheduler and streaming interface share the Reader pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .adapters.reader import PreparedEvidence
from .retrieval.dynamic_budget import route_question
from .retrieval.schema_entity import slots, names
from .scheduler import Question


@dataclass
class StreamingQuerySession:
    reader: Any
    user_id: str
    snapshot: Sequence[dict]
    cutoff_record_date: str | None = None
    partial_transcript: str = ""
    final_transcript: str | None = None
    preview: PreparedEvidence | None = None
    final_evidence: PreparedEvidence | None = None
    stage: str = "LISTENING"

    def listen(self, partial_transcript: str) -> dict:
        """Supply the latest cumulative partial ASR transcript (revisions allowed)."""
        if self.final_transcript is not None:
            raise RuntimeError("speech tail already finalized")
        if not partial_transcript.strip():
            raise ValueError("nonempty partial transcript required")
        self.partial_transcript = partial_transcript.strip()
        self.preview = None
        self.stage = "LISTENING"
        return self.cues()

    def listen_audio(self, wav_chunk: bytes, asr_adapter: Any) -> dict:
        """Recognize a complete short WAV chunk and append its partial words.

        The frontend sets audio chunk boundaries; the HTTP ASR service
        processes each self-contained WAV chunk.
        """
        text = asr_adapter.transcribe_wav_bytes(wav_chunk)
        combined = (self.partial_transcript + " " + text).strip()
        return self.listen(combined)

    def cues(self) -> dict:
        """Extract provisional source-derived intent, domain, and entity cues."""
        return {
            "stage": self.stage,
            "domains": list(slots(self.partial_transcript)),
            "entities": list(names(self.partial_transcript)),
            "intent": route_question(self.partial_transcript) if self.partial_transcript else "GENERAL",
        }

    def anticipate(self) -> PreparedEvidence:
        """Prepare candidate evidence from partial or finalized speech.

        Early calls are speculative; after Speech Tail, the finalized candidate
        context is authoritative and can be reused by Searching.
        """
        text = self.final_transcript or self.partial_transcript
        if not text:
            raise ValueError("LISTENING_REQUIRED")
        prepared = self.reader.prepare_evidence(
            text, self.user_id, self.snapshot,
            cutoff_record_date=self.cutoff_record_date)
        if self.final_transcript is None:
            self.preview = prepared
        else:
            self.final_evidence = prepared
        self.stage = "ANTICIPATION"
        return prepared

    def speech_tail(self, final_segment: str) -> str:
        """Incorporate the final segment into the resolved full utterance."""
        if self.final_transcript is not None:
            raise RuntimeError("speech tail already finalized")
        if not final_segment.strip():
            raise ValueError("nonempty speech tail required")
        self.final_transcript = (self.partial_transcript + " " + final_segment).strip()
        self.stage = "SPEECH_TAIL"
        return self.final_transcript

    def speech_tail_audio(self, wav_chunk: bytes, asr_adapter: Any) -> str:
        return self.speech_tail(asr_adapter.transcribe_wav_bytes(wav_chunk))

    def search(self) -> PreparedEvidence:
        """Finalize trajectory retrieval using the complete resolved query."""
        if self.final_transcript is None:
            raise RuntimeError("SPEECH_TAIL_REQUIRED")
        if self.final_evidence is None:
            if self.preview is not None and self.preview.question == self.final_transcript:
                self.final_evidence = self.preview
            else:
                self.final_evidence = self.reader.prepare_evidence(
                    self.final_transcript, self.user_id, self.snapshot,
                    cutoff_record_date=self.cutoff_record_date)
        self.stage = "SEARCHING"
        return self.final_evidence

    def answer(self, question_id: str) -> dict:
        """Generate from the single final evidence context, without re-retrieval."""
        evidence = self.search()
        question = Question(self.user_id, question_id, self.final_transcript,
                            self.cutoff_record_date)
        return self.reader.answer_prepared(question, evidence)
