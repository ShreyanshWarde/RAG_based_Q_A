import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    _model = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def model(self) -> SentenceTransformer:
        if self.__class__._model is None:
            with self.__class__._lock:
                if self.__class__._model is None:
                    self.__class__._model = SentenceTransformer(self.settings.embedding_model_name)
        return self.__class__._model

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
