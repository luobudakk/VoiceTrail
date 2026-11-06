'''Composition of independent VoiceTrail ASR, Writer, Indexer and Reader stages.'''
from __future__ import annotations
from .asr import ASRAdapter
from .writer import WriterAdapter
from .indexing import IndexAdapter
from .reader import ReaderAdapter
from ..config import Settings

class LiveStages(ASRAdapter, WriterAdapter, IndexAdapter, ReaderAdapter):
    def __init__(self, settings: Settings) -> None:
        import requests
        from sentence_transformers import SentenceTransformer

        self.requests = requests
        self.asr_url = settings.asr_url
        self.asr_token = settings.asr_api_key
        self.writer_url = settings.writer_url
        self.reader_url = settings.reader_url
        self.writer_model = settings.writer_model
        self.reader_model_name = settings.reader_model_name
        self.writer_headers = (
            {'Authorization': 'Bearer ' + settings.writer_api_key}
            if settings.writer_api_key else {})
        self.reader_headers = (
            {'Authorization': 'Bearer ' + settings.reader_api_key}
            if settings.reader_api_key else {})
        self.reader_model_path = settings.reader_model_path
        self.reader_device = settings.reader_device
        self.reader_cpu_threads = settings.reader_cpu_threads
        self.reader_memory_fraction = settings.reader_memory_fraction
        self.e5 = SentenceTransformer(settings.e5_model, device='cpu')
        self.reader_model = None
        self.reader_tokenizer = None
