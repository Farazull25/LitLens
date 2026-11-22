"""Local CPU embedding provider using BAAI/bge-small-en-v1.5 with deterministic hash fallback for offline tests."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.observability.logger import get_logger

TOKEN_SPLIT_RE = re.compile(r"[a-z0-9_]+")
HASH_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "which", "when", "where", "who",
    "how", "into", "than", "then", "also", "can", "will", "does", "used", "many",
    "much", "why", "under", "across", "during", "between",
}


class LocalEmbedder:
    """Produces L2-normalized 384-dim dense embeddings on CPU."""

    def __init__(self, model_name: str | None = None, dim: int = 384, force_hash: bool = False) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.dim = dim
        self.force_hash = force_hash or (settings.environment == "test")
        self._st_model = None

    def _load_model(self) -> None:
        if self.force_hash or self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name, device="cpu")
            get_logger().info("embedding_model_loaded", model=self.model_name)
        except Exception as exc:
            get_logger().warning("sentence_transformer_fallback_to_hash", error=str(exc))
            self.force_hash = True

    def _hash_embed(self, texts: list[str]) -> np.ndarray:
        """Fast deterministic signed feature-hashing embedding (L2 normalized) for offline/test environments."""
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = [t for t in TOKEN_SPLIT_RE.findall(text.lower()) if t not in HASH_STOPWORDS]
            if not tokens:
                matrix[row_idx, 0] = 1.0
                continue
            # Include unigrams and bigrams for strong semantic discrimination in fallback mode
            features = list(tokens) + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            for feat in features:
                digest = hashlib.md5(feat.encode("utf-8"), usedforsecurity=False).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                matrix[row_idx, bucket] += sign
            norm = float(np.linalg.norm(matrix[row_idx]))
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode a list of texts into L2-normalized float32 embeddings of shape (N, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._load_model()
        if self._st_model is not None and not self.force_hash:
            # BGE models benefit from query instruction prefix for retrieval queries
            prefixed = (
                [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
                if is_query and "bge" in self.model_name.lower()
                else texts
            )
            vecs = self._st_model.encode(
                prefixed,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)

        return self._hash_embed(texts)


@lru_cache(maxsize=2)
def get_embedder(force_hash: bool = False) -> LocalEmbedder:
    """Return singleton embedder."""
    return LocalEmbedder(force_hash=force_hash)
