"""Qwen2.5-Omni WAV transcription, with a chunk-oriented streaming entrypoint."""
from pathlib import Path
import base64
from ..scheduler import AudioTurn


class ASRAdapter:
    def transcribe_wav_bytes(self, wav: bytes) -> str:
        """Transcribe a complete WAV or self-contained short WAV chunk."""
        if not 0 < len(wav) < 5_000_000:
            raise ValueError("missing or unexpectedly large input WAV")
        response = self.requests.post(
            self.asr_url.rstrip("/") + "/v1/audio/analyze",
            headers={"Authorization": "Bearer " + self.asr_token},
            json={
                "audio_base64": base64.b64encode(wav).decode("ascii"),
                "prompt": "Transcribe exactly what is spoken in this audio in its original language. "
                          "Return only the spoken words without descriptions.",
                "max_new_tokens": 192,
            },
            timeout=160,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("model") != "Qwen2.5-Omni-7B":
            raise RuntimeError("incorrect Omni response model")
        return str(payload["text"]).strip()

    def asr(self, turn: AudioTurn) -> str:
        return self.transcribe_wav_bytes(Path(turn.audio_path).read_bytes())
