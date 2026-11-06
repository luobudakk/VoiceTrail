'''Environment-driven configuration for public VoiceTrail deployments.'''
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

def _required(key: str) -> str:
    value = os.getenv(key, '').strip()
    if not value:
        raise ValueError(f'Missing required environment variable: {key}')
    return value

def _optional_path(key: str) -> Path | None:
    value = os.getenv(key, '').strip()
    return Path(value).expanduser() if value else None

@dataclass(frozen=True)
class Settings:
    asr_url: str
    asr_api_key: str
    writer_url: str
    e5_model: str
    reader_url: str | None = None
    reader_model_path: Path | None = None
    writer_model: str = 'qwen3'
    reader_model_name: str = 'qwen3'
    writer_api_key: str | None = None
    reader_api_key: str | None = None
    reader_device: str = 'cuda:0'
    reader_cpu_threads: int = 4
    reader_memory_fraction: float = 0.45

    @classmethod
    def from_env(cls) -> 'Settings':
        url = os.getenv('VOICETRAIL_READER_URL', '').strip() or None
        model_path = _optional_path('VOICETRAIL_READER_MODEL_PATH')
        if url is None and model_path is None:
            raise ValueError(
                'Set VOICETRAIL_READER_URL or VOICETRAIL_READER_MODEL_PATH')
        fraction = float(os.getenv('VOICETRAIL_READER_GPU_FRACTION', '0.45'))
        if not 0 < fraction <= 1:
            raise ValueError('VOICETRAIL_READER_GPU_FRACTION must be in (0,1]')
        return cls(
            asr_url=_required('VOICETRAIL_ASR_URL'),
            asr_api_key=_required('VOICETRAIL_ASR_API_KEY'),
            writer_url=_required('VOICETRAIL_WRITER_URL'),
            e5_model=os.getenv('VOICETRAIL_E5_MODEL', 'intfloat/multilingual-e5-large'),
            reader_url=url, reader_model_path=model_path,
            writer_model=os.getenv('VOICETRAIL_WRITER_MODEL', 'qwen3'),
            reader_model_name=os.getenv('VOICETRAIL_READER_MODEL_NAME', 'qwen3'),
            writer_api_key=os.getenv('VOICETRAIL_WRITER_API_KEY') or None,
            reader_api_key=os.getenv('VOICETRAIL_READER_API_KEY') or None,
            reader_device=os.getenv('VOICETRAIL_READER_DEVICE', 'cuda:0'),
            reader_cpu_threads=int(os.getenv('VOICETRAIL_READER_CPU_THREADS','4')),
            reader_memory_fraction=fraction,
        )
