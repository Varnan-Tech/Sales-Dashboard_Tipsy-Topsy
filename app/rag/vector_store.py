from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

VECTOR_BACKEND = os.getenv("VECTOR_STORE", "chroma").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")


class VectorStore:
	"""Abstract vector store wrapper supporting namespaces by dataset_id."""

	def create_collection(self, dataset_id: str) -> None:
		raise NotImplementedError

	def upsert(self, dataset_id: str, ids: List[str], vectors: np.ndarray, metadatas: List[Dict]) -> None:
		raise NotImplementedError

	def query(self, dataset_id: str, query_vector: np.ndarray, top_k: int = 8) -> List[Dict]:
		raise NotImplementedError

	def delete_collection(self, dataset_id: str) -> None:
		raise NotImplementedError

	def persist(self) -> None:
		pass

	def load_collection(self, dataset_id: str) -> None:
		pass


class ChromaStore(VectorStore):
	def __init__(self) -> None:
		import chromadb

		# Use new Chroma client API
		self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
		self._collections: Dict[str, object] = {}

	def create_collection(self, dataset_id: str) -> None:
		if dataset_id in self._collections:
			return
		self._collections[dataset_id] = self._client.get_or_create_collection(dataset_id)

	def upsert(self, dataset_id: str, ids: List[str], vectors: np.ndarray, metadatas: List[Dict], documents: Optional[List[str]] = None) -> None:
		self.create_collection(dataset_id)
		col = self._collections[dataset_id]
		# Extract documents from metadata if not provided separately
		if documents is None:
			documents = [meta.get("document", "") for meta in metadatas]
		col.upsert(ids=ids, embeddings=vectors.tolist(), metadatas=metadatas, documents=documents)

	def query(self, dataset_id: str, query_vector: np.ndarray, top_k: int = 8) -> List[Dict]:
		self.create_collection(dataset_id)
		col = self._collections[dataset_id]
		res = col.query(query_embeddings=[query_vector.tolist()], n_results=top_k, include=["documents", "metadatas", "distances"])
		out: List[Dict] = []
		for i in range(len(res["ids"][0])):
			out.append({
				"id": res["ids"][0][i],
				"metadata": res["metadatas"][0][i],
				"content": res["documents"][0][i] if "documents" in res else "",
				"distance": res.get("distances", [[None]])[0][i],
				"score": 1.0 - (res.get("distances", [[0]])[0][i] or 0) if res.get("distances") else 0.0,  # Convert distance to similarity score
			})
		return out

	def delete_collection(self, dataset_id: str) -> None:
		if dataset_id in self._collections:
			self._client.delete_collection(dataset_id)
			self._collections.pop(dataset_id, None)

	def persist(self) -> None:
		# Chroma persists automatically when configured
		pass


class FaissStore(VectorStore):
	def __init__(self) -> None:
		import faiss  # type: ignore

		self._faiss = faiss
		self._index: Optional[object] = None
		self._id_to_meta: Dict[str, Dict] = {}

	def create_collection(self, dataset_id: str) -> None:
		# Single index for simplicity; in production, separate by namespace
		if self._index is None:
			self._index = self._faiss.IndexFlatIP(384)

	def upsert(self, dataset_id: str, ids: List[str], vectors: np.ndarray, metadatas: List[Dict]) -> None:
		self.create_collection(dataset_id)
		# Idempotent upsert: FAISS lacks native upsert; simplistic replace
		for i, vid in enumerate(ids):
			self._id_to_meta[vid] = metadatas[i]
		self._index.add(vectors.astype("float32"))

	def query(self, dataset_id: str, query_vector: np.ndarray, top_k: int = 8) -> List[Dict]:
		D, I = self._index.search(query_vector.reshape(1, -1).astype("float32"), top_k)  # type: ignore
		out: List[Dict] = []
		for dist, idx in zip(D[0], I[0]):
			if idx == -1:
				continue
			out.append({"id": str(idx), "metadata": {}, "distance": float(dist)})
		return out

	def delete_collection(self, dataset_id: str) -> None:
		self._index = None
		self._id_to_meta.clear()


def get_vector_store() -> VectorStore:
	if VECTOR_BACKEND == "faiss":
		return FaissStore()
	return ChromaStore()
