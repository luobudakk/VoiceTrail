"""Audio transcription adapter for an HTTP speech model."""
from pathlib import Path
import base64
from ..scheduler import AudioTurn

class ASRAdapter:
    def asr(self, turn: AudioTurn) -> str:
        wav = Path(turn.audio_path).read_bytes()
        if not 0 < len(wav) < 5_000_000:
            raise ValueError("missing or unexpectedly large input WAV")
        response = self.requests.post(
            self.asr_url.rstrip("/") + "/v1/audio/analyze",
            headers={"Authorization": "Bearer " + self.asr_token},
            json={
                "audio_base64": base64.b64encode(wav).decode("ascii"),
                "prompt": "Transcribe exactly what is spoken in this English audio. "
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
