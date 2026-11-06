"""Semantic indexing using multilingual E5 embeddings."""
from typing import Any
from ..scheduler import Session

class IndexAdapter:
    def indexer(self, session: Session, memory: dict) -> dict[str, Any]:
        items = [
            *[(r["turn_id"], r["text"]) for r in memory["raw"]],
            *[(f["fact_id"], f["text"]) for f in memory["facts"]],
            *[(o["obs_id"], ", ".join(o["emotion"]) + " " + o["emotion_quote"])
              for o in memory["emotions"]],
        ]
        vectors = self.e5.encode(
            ["passage: " + text for _, text in items],
            normalize_embeddings=True, batch_size=8, show_progress_bar=False,
        )
        return {
            "user_id": session.user_id, "session_id": session.session_id,
            "date": session.date,
            "raw": memory["raw"], "facts": memory["facts"],
            "emotions": memory["emotions"],
            "vectors": {key: vec.tolist() for (key, _), vec in zip(items, vectors)},
        }
