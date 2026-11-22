"""FAISS local vector index with S3 snapshot synchronization and duplicate filtering."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from threading import Lock

import numpy as np

from app.config import get_settings
from app.observability.logger import get_logger
from app.rag.chunker import PaperChunk
from app.rag.embeddings import LocalEmbedder, get_embedder
from app.rag.reranker import RetrievedChunk, fuse_dense_and_bm25
from app.storage.s3 import S3Store

try:
    import faiss  # type: ignore[import-untyped]

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FAISSVectorStore:
    """In-memory FAISS IndexFlatIP (cosine similarity) backed by local disk & S3 snapshots."""

    def __init__(
        self,
        embedder: LocalEmbedder | None = None,
        index_dir: str | Path | None = None,
        s3_store: S3Store | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or get_embedder()
        self.dim = self.embedder.dim
        self.index_dir = Path(index_dir or settings.local_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.s3_store = s3_store or S3Store()
        self._lock = Lock()

        self.chunks: list[PaperChunk] = []
        self._chunk_ids: set[str] = set()
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._faiss_index = faiss.IndexFlatIP(self.dim) if HAS_FAISS else None

    @property
    def index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "chunks_meta.json.gz"

    def add_chunks(self, new_chunks: list[PaperChunk], sync_s3: bool = False) -> int:
        """Embed and add non-duplicate chunks to the FAISS index."""
        unique = [c for c in new_chunks if c.chunk_id not in self._chunk_ids]
        if not unique:
            return 0

        texts = [f"{c.title} - {c.section}: {c.text}" for c in unique]
        embeddings = self.embedder.encode(texts, is_query=False)

        with self._lock:
            for c in unique:
                self.chunks.append(c)
                self._chunk_ids.add(c.chunk_id)

            self._vectors = (
                np.vstack([self._vectors, embeddings])
                if len(self._vectors) > 0
                else embeddings
            )
            if self._faiss_index is not None:
                self._faiss_index.add(embeddings)

            self.save_local()
            if sync_s3:
                self.s3_store.upload_faiss_snapshot(self.index_path, self.meta_path)

        return len(unique)

    def search(
        self,
        query: str,
        top_k: int = 5,
        arxiv_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid FAISS dense + BM25 search with optional paper filtering."""
        if not self.chunks:
            return []

        q_vec = self.embedder.encode([query], is_query=True)

        with self._lock:
            if arxiv_ids:
                allowed = set(arxiv_ids)
                candidate_indices = [
                    i for i, c in enumerate(self.chunks) if c.arxiv_id in allowed
                ]
            else:
                candidate_indices = list(range(len(self.chunks)))

            if not candidate_indices:
                return []

            candidate_chunks = [self.chunks[i] for i in candidate_indices]
            candidate_vecs = self._vectors[candidate_indices]
            dense_scores = np.dot(candidate_vecs, q_vec[0]).tolist()

        return fuse_dense_and_bm25(
            query=query,
            chunks=candidate_chunks,
            dense_scores=dense_scores,
            top_k=top_k,
        )

    def list_papers(self) -> list[dict[str, str]]:
        """Return unique metadata of all indexed arXiv papers."""
        seen: dict[str, dict[str, str]] = {}
        for c in self.chunks:
            if c.arxiv_id not in seen:
                seen[c.arxiv_id] = {
                    "arxiv_id": c.arxiv_id,
                    "title": c.title,
                    "authors": ", ".join(c.authors),
                    "published": c.published,
                }
        return list(seen.values())

    def save_local(self) -> None:
        """Persist FAISS index and metadata locally."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        meta_payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self._vectors.tolist(),
        }
        self.meta_path.write_bytes(
            gzip.compress(json.dumps(meta_payload, ensure_ascii=False).encode("utf-8"))
        )
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self.index_path))

    def load_local_or_s3(self) -> bool:
        """Load index from local disk or hydrate from S3 snapshot."""
        if not self.meta_path.exists():
            self.s3_store.download_faiss_snapshot(self.index_path, self.meta_path)

        if not self.meta_path.exists():
            return False

        try:
            raw = json.loads(gzip.decompress(self.meta_path.read_bytes()).decode("utf-8"))
            loaded_chunks = [PaperChunk(**item) for item in raw.get("chunks", [])]
            loaded_vecs = np.asarray(raw.get("vectors", []), dtype=np.float32)
            if len(loaded_chunks) == 0:
                return False

            with self._lock:
                self.chunks = loaded_chunks
                self._chunk_ids = {c.chunk_id for c in loaded_chunks}
                self._vectors = loaded_vecs
                if HAS_FAISS:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                    self._faiss_index.add(self._vectors)
            get_logger().info("faiss_index_hydrated", chunk_count=len(self.chunks))
            return True
        except Exception as exc:
            get_logger().warning("faiss_index_load_failed", error=str(exc))
            return False
