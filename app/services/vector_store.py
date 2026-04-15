import json
import threading
from pathlib import Path

import faiss
import numpy as np

from app.config import get_settings


class VectorStore:
    _shared_index = None
    _shared_metadata: list[dict[str, str]] | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_path: Path = self.settings.vector_store_dir / "index.faiss"
        self.metadata_path: Path = self.settings.vector_store_dir / "chunks.json"
        self.top_k = self.settings.top_k
        self.similarity_threshold = self.settings.min_similarity_threshold
        self._load_if_needed()

    def _load_if_needed(self) -> None:
        with self.__class__._lock:
            if self.__class__._shared_index is not None and self.__class__._shared_metadata is not None:
                return

            if self.index_path.exists() and self.metadata_path.exists():
                self.__class__._shared_index = faiss.read_index(str(self.index_path))
                self.__class__._shared_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                return

            self.__class__._shared_index = None
            self.__class__._shared_metadata = []

    @property
    def index(self):
        return self.__class__._shared_index

    @property
    def metadata(self) -> list[dict[str, str]]:
        return self.__class__._shared_metadata

    def add_embeddings(self, chunks: list[str], embeddings: np.ndarray, source_file: str) -> None:
        with self.__class__._lock:
            if embeddings.ndim != 2:
                raise ValueError("Embeddings must be a 2D numpy array.")

            if self.index is None:
                dimension = embeddings.shape[1]
                self.__class__._shared_index = faiss.IndexFlatIP(dimension)

            self.index.add(embeddings)

            for index, chunk in enumerate(chunks):
                self.metadata.append(
                    {
                        "source_file": source_file,
                        "chunk_id": f"{source_file}:{index}",
                        "text": chunk,
                    }
                )

            self._persist()

    def search(self, query_embedding: np.ndarray) -> list[dict[str, str | float]]:
        with self.__class__._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            query = np.expand_dims(query_embedding.astype("float32"), axis=0)
            scores, indices = self.index.search(query, self.top_k)
            matches: list[dict[str, str | float]] = []

            for score, index in zip(scores[0], indices[0], strict=False):
                if index < 0 or score < self.similarity_threshold:
                    continue

                item = self.metadata[index]
                matches.append(
                    {
                        "score": float(score),
                        "source_file": item["source_file"],
                        "text": item["text"],
                    }
                )

            return matches

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
