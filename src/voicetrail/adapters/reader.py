"""VoiceTrail V7 retrieval and evidence-grounded Reader adapter."""
import os
import time
from typing import Any
from ..scheduler import Question

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

    def answer(self, question: Question, snapshot: tuple[dict, ...]) -> dict[str, Any]:
        import numpy as np
        from ..retrieval import assemble_query, PIPELINE_POLICY

        if not snapshot or any(s["user_id"] != question.user_id for s in snapshot):
            raise ValueError("cross-user memory contamination")
        raw = [r for s in snapshot for r in s["raw"]]
        facts = [f for s in snapshot for f in s["facts"]]
        obs = [o for s in snapshot for o in s["emotions"]]
        vectors = {key: val for s in snapshot for key, val in s["vectors"].items()}
        v = self.e5.encode(["query: " + question.text],
                           normalize_embeddings=True, show_progress_bar=False)[0]
        def scores(items, key):
            return {row[key]: float(np.dot(v, np.asarray(vectors[row[key]], dtype=np.float32)))
                    for row in items}
        result = assemble_query(
            question.text, user_id=question.user_id,
            cutoff_record_date=max(s["date"] for s in snapshot),
            raw_pool=raw, fact_pool=facts, emotion_pool=obs,
            raw_scores=scores(raw, "turn_id"), fact_scores=scores(facts, "fact_id"),
            emotion_scores=scores(obs, "obs_id"),
            schema_enabled=True, trail_enabled=True, emotion_enabled=True,
            neighbor_enabled=True, fact_enabled=True, mode="dynamic",
        )
        if result.receipt["pipeline"] != PIPELINE_POLICY:
            raise ValueError("wrong VoiceTrail V7 version")
        prompt = (
            "QUESTION:\n" + question.text
            + "\nAS_OF_RECORD_DATE: " + max(s["date"] for s in snapshot)
            + "\nHISTORICAL EVIDENCE:\n" + result.context
        )
        system = (
            "Answer only using historical evidence and respect source roles, dates "
            "and person identity. Do not ascribe another person's feelings to seeker. "
            "Do not invent unobserved emotions, causal links or future states. "
            "Give a short direct answer; say Insufficient information when unsupported."
        )
        if self.reader_url:
            # HTTP Reader can reuse an existing model service.
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
                    result.context, add_special_tokens=False))
                    if self.reader_tokenizer is not None else None),
                "total_prompt_tokens": int(data.get("usage", {}).get("prompt_tokens", 0)),
                "output_tokens": int(data.get("usage", {}).get("completion_tokens", 0)),
                "generation_seconds": round(time.monotonic() - begun, 4),
                "selected_evidence": result.receipt["budget_receipt"]["selected_count"],
                "trail_enabled": result.receipt["trail"]["enabled"],
                "evidence_source_ids": result.receipt["budget_receipt"],
                "input_session_count": len(snapshot),
                "model_server_batch_size": data.get("_runtime", {}).get("batch_size"),
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
                result.context, add_special_tokens=False)),
            "total_prompt_tokens": int(count),
            "output_tokens": int(output.shape[-1] - count),
            "generation_seconds": round(time.monotonic() - begun, 4),
            "selected_evidence": result.receipt["budget_receipt"]["selected_count"],
            "trail_enabled": result.receipt["trail"]["enabled"],
            "evidence_source_ids": result.receipt["budget_receipt"],
            "input_session_count": len(snapshot),
        }
