from __future__ import annotations

import os
from typing import List

import numpy as np

_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "256") or 256)


class Embeddings:
	def __init__(self, model_name: str = _MODEL_NAME) -> None:
		from sentence_transformers import SentenceTransformer

		self._model = SentenceTransformer(model_name)

	def embed_batch(self, texts: List[str], batch_size: int = _BATCH_SIZE) -> np.ndarray:
		embs = self._model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
		return np.asarray(embs, dtype=np.float32)
