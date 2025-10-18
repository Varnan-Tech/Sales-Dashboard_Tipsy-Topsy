from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

DEFAULT_MAX_SYNC_ROWS = int(os.getenv("MAX_SYNC_INDEX_ROWS", "10000") or 10000)
DEFAULT_TOP_N_COLS = int(os.getenv("INDEX_MAX_COLUMNS", "25") or 25)


@dataclass
class IngestResult:
	rows_indexed: int
	duration_s: float
	dataset_id: str
	summary_doc_id: Optional[str] = None


def _select_columns(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
	if df.shape[1] <= top_n:
		return df
	return df.iloc[:, :top_n]


def _row_to_text(row: pd.Series) -> str:
	parts = [f"{k}: {row[k]}" for k in row.index]
	return " | ".join(parts)


def generate_dataset_summary(dataset_id: str, df: pd.DataFrame) -> str:
	lines = [f"Dataset: {dataset_id}"]
	lines.append(f"Rows: {len(df):,} Cols: {df.shape[1]}")
	lines.append("Columns: " + ", ".join(map(str, df.columns[:50])))
	return "\n".join(lines)


def iter_documents(
	dataset_id: str,
	df: pd.DataFrame,
	file_name: str,
	primary_key: Optional[str] = None,
	max_columns: int = DEFAULT_TOP_N_COLS,
) -> Iterable[Tuple[str, str, Dict]]:
	"""Yield (doc_id, text, metadata) per row."""
	df2 = _select_columns(df, max_columns)
	for idx, row in df2.iterrows():
		pk_val = row.get(primary_key) if primary_key and primary_key in row else None
		doc_id = f"{dataset_id}-{idx}"
		text = _row_to_text(row)
		meta = {
			"dataset_id": dataset_id,
			"file_name": file_name,
			"row_index": int(idx),
			"primary_key": pk_val,
			"row_range": f"{idx}-{idx}",
			"ingest_timestamp": int(time.time()),
		}
		yield doc_id, text, meta


def index_dataframe(
	dataset_id: str,
	df: pd.DataFrame,
	file_name: str,
	vector_store,
	embeddings,
	primary_key: Optional[str] = None,
	max_sync_rows: int = DEFAULT_MAX_SYNC_ROWS,
) -> IngestResult:
	"""Index dataframe rows synchronously up to max_sync_rows."""
	start = time.time()
	rows = 0
	batch_ids: List[str] = []
	batch_texts: List[str] = []
	batch_metadatas: List[Dict] = []

	for doc_id, text, meta in iter_documents(dataset_id, df.head(max_sync_rows), file_name, primary_key):
		batch_ids.append(doc_id)
		batch_texts.append(text)
		batch_metadatas.append(meta)
		rows += 1

	if batch_ids:
		vectors = embeddings.embed_batch(batch_texts)
		vector_store.upsert(dataset_id, batch_ids, vectors, batch_metadatas)

	return IngestResult(rows_indexed=rows, duration_s=time.time() - start, dataset_id=dataset_id)
