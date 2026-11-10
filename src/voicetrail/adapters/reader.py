"""VoiceTrail source-grounded retrieval and Reader adapter.

Query preparation is a first-class step, permitting anticipation on partial
transcripts, a final Speech Tail update, and independent latency accounting.
"""
from dataclasses import dataclass
from datetime import date
import time
from typing import Any, Sequence
from ..scheduler import Question


@dataclass(frozen=True)
class PreparedEvidence:
    question: str
    user_id: str
    cutoff_record_date: str
    context: str
    receipt: dict[str, Any]
    retrieval_latency_ms: float
    query_embedding_ms: float
    eligible_session_count: int


class ReaderAdapter:
    def _load_reader(self):
        if self.reader_model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(self.reader_cpu_threads)
        if self.reader_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA Reader requires a GPU")
            torch.cuda.set_per_process_memory_fraction(self.reader_memory_fraction, device=0)
        model_dir = self.reader_model_path
        if not model_dir:
            raise ValueError("Set VOICETRAIL_READER_MODEL_PATH for local Reader inference")
        self.reader_tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True)
        self.reader_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16 if self.reader_device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
            local_files_only=True).to(self.reader_device).eval()

    def prepare_evidence(self, query_text: str, user_id: str,
                         snapshot: Sequence[dict], *,
                         cutoff_record_date: str | None = None) -> PreparedEvidence:
        """Run deterministic memory retrieval before answer generation.

        The cutoff is per question. Exclude future indexed sessions before
        scoring; never let a later session leak into an earlier snapshot.
        """
        import numpy as np
        from ..retrieval import assemble_query
        if not query_text.strip() or not user_id or not snapshot:
            raise ValueError("query, user and snapshot required")
        if any(s["user_id"] != user_id for s in snapshot):
            raise ValueError("cross-user memory contamination")
        cutoff = cutoff_record_date or max(s["date"] for s in snapshot)
        for value in [cutoff, *(s["date"] for s in snapshot)]:
            if not isinstance(value, str) or len(value) != 10:
                raise ValueError("INVALID_RECORD_DATE")
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("INVALID_RECORD_DATE")
            except ValueError as exc:
                raise ValueError("INVALID_RECORD_DATE") from exc
        legal = [s for s in snapshot if s["date"] <= cutoff]
        if not legal:
            return PreparedEvidence(query_text, user_id, cutoff, "", {},
                                    0.0, 0.0, 0)
        begun = time.perf_counter()
        raw = [r for s in legal for r in s["raw"]]
        facts = [f for s in legal for f in s["facts"]]
        obs = [o for s in legal for o in s["emotions"]]
        vectors = {key: val for s in legal for key, val in s["vectors"].items()}
        embedding_start = time.perf_counter()
        query_vec = self.e5.encode(["query: " + query_text],
                                   normalize_embeddings=True, show_progress_bar=False)[0]
        embedding_ms = (time.perf_counter() - embedding_start) * 1000
        def scores(items, key):
            return {row[key]: float(np.dot(query_vec, np.asarray(
                vectors[row[key]], dtype=np.float32))) for row in items}
        result = assemble_query(
            query_text, user_id=user_id, cutoff_record_date=cutoff,
            raw_pool=raw, fact_pool=facts, emotion_pool=obs,
            raw_scores=scores(raw, "turn_id"), fact_scores=scores(facts, "fact_id"),
            emotion_scores=scores(obs, "obs_id"),
            schema_enabled=True, trail_enabled=True, emotion_enabled=True,
            neighbor_enabled=True, fact_enabled=True, mode="dynamic",
        )
        return PreparedEvidence(
            query_text, user_id, cutoff, result.context, result.receipt,
            round((time.perf_counter() - begun) * 1000, 3),
            round(embedding_ms, 3), len(legal))

    def answer_prepared(self, question: Question,
                        prepared: PreparedEvidence) -> dict[str, Any]:
        """Generate from final prepared evidence; do not redo retrieval."""
        from ..retrieval import PIPELINE_POLICY
        if question.user_id != prepared.user_id or question.text != prepared.question:
            raise ValueError("PREPARED_QUERY_IDENTITY_MISMATCH")
        if question.cutoff_record_date and question.cutoff_record_date != prepared.cutoff_record_date:
            raise ValueError("PREPARED_QUERY_CUTOFF_MISMATCH")
        if not prepared.eligible_session_count:
            return {
                "answer": "Insufficient information", "reader_model": self.reader_model_name,
                "reader_mode": "no_eligible_history", "memory_token_count": 0,
                "total_prompt_tokens": 0, "output_tokens": 0,
                "generation_seconds": 0.0, "retrieval_latency_ms": 0.0,
                "query_embedding_ms": 0.0, "selected_evidence": [0, 0, 0],
                "trail_enabled": True, "evidence_source_ids": {},
                "input_session_count": 0,
            }
        if prepared.receipt["pipeline"] != PIPELINE_POLICY:
            raise ValueError("wrong VoiceTrail retrieval version")
        prompt = (
            "QUESTION:\n" + question.text
            + "\nAS_OF_RECORD_DATE: " + prepared.cutoff_record_date
            + "\nHISTORICAL EVIDENCE:\n" + prepared.context
        )
        system = (
            "Answer only using historical evidence and respect source roles, dates "
            "and person identity. Do not ascribe another person's feelings to seeker. "
            "Do not invent unobserved emotions, causal links or future states. "
            "Distinguish experienced emotions from hypothetical or intended feelings. "
            "Give a short direct answer; say Insufficient information when unsupported."
        )
        metadata = {
            "retrieval_latency_ms": prepared.retrieval_latency_ms,
            "query_embedding_ms": prepared.query_embedding_ms,
            "selected_evidence": prepared.receipt["budget_receipt"]["selected_count"],
            "trail_enabled": prepared.receipt["trail"]["enabled"],
            "evidence_source_ids": prepared.receipt["budget_receipt"],
            "input_session_count": prepared.eligible_session_count,
        }
        if self.reader_url:
            from transformers import AutoTokenizer
            if self.reader_tokenizer is None and self.reader_model_path:
                self.reader_tokenizer = AutoTokenizer.from_pretrained(
                    str(self.reader_model_path), local_files_only=True)
            begun = time.monotonic()
            reply = self.requests.post(
                self.reader_url,
                headers=self.reader_headers,
                json={
                    "model": self.reader_model_name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 128,
                    "stream": False,
                },
                timeout=160,
            )
            reply.raise_for_status()
            data = reply.json()
            return {
                "answer": str(data["choices"][0]["message"]["content"]).strip(),
                "reader_model": str(data.get("model", "Qwen3-8B")),
                "reader_mode": "http_model",
                "memory_token_count": (len(self.reader_tokenizer.encode(
                    prepared.context, add_special_tokens=False))
                    if self.reader_tokenizer is not None else None),
                "total_prompt_tokens": int(data.get("usage", {}).get("prompt_tokens", 0)),
                "output_tokens": int(data.get("usage", {}).get("completion_tokens", 0)),
                "generation_seconds": round(time.monotonic() - begun, 4),
                "model_server_batch_size": data.get("_runtime", {}).get("batch_size"),
                **metadata,
            }
        import torch
        self._load_reader()
        ids = self.reader_tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        )
        if hasattr(ids, "get"):
            ids = ids["input_ids"]
        count = ids.shape[-1]
        begun = time.monotonic()
        with torch.inference_mode():
            output = self.reader_model.generate(
                ids.to(self.reader_device), do_sample=False, max_new_tokens=128,
                use_cache=True, pad_token_id=self.reader_tokenizer.eos_token_id,
            )
        answer = self.reader_tokenizer.decode(
            output[0, count:], skip_special_tokens=True).strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[-1].strip()
        return {
            "answer": answer, "reader_model": self.reader_model_name,
            "memory_token_count": len(self.reader_tokenizer.encode(
                prepared.context, add_special_tokens=False)),
            "total_prompt_tokens": int(count),
            "output_tokens": int(output.shape[-1] - count),
            "generation_seconds": round(time.monotonic() - begun, 4),
            **metadata,
        }

    def answer(self, question: Question, snapshot: tuple[dict, ...]) -> dict[str, Any]:
        prepared = self.prepare_evidence(
            question.text, question.user_id, snapshot,
            cutoff_record_date=question.cutoff_record_date)
        return self.answer_prepared(question, prepared)
