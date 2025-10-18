from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import pandas as pd
from rq import get_current_job
from redis import Redis

from .embeddings import Embeddings
from .indexer import index_dataframe
from .vector_store import get_vector_store

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _publish_progress(r: Redis, dataset_id: str, pct: float, msg: str) -> None:
	chan = f"INDEX_PROGRESS:{dataset_id}"
	r.publish(chan, json.dumps({"progress": pct, "message": msg, "ts": int(time.time())}))


def index_dataset_job(dataset_id: str, file_path: str, file_name: str, primary_key: str | None = None) -> Dict[str, Any]:
	r = Redis.from_url(REDIS_URL)
	_publish_progress(r, dataset_id, 0.0, "Starting indexing...")
	job = get_current_job()
	job.meta["dataset_id"] = dataset_id
	job.save_meta()

	# Load CSV for now (extend to xlsx later)
	df = pd.read_csv(file_path)
	vec = get_vector_store()
	emb = Embeddings()
	_publish_progress(r, dataset_id, 0.1, "Loaded file; computing embeddings...")
	res = index_dataframe(dataset_id, df, file_name, vec, emb, primary_key=primary_key)
	_publish_progress(r, dataset_id, 1.0, f"Indexed {res.rows_indexed} rows in {res.duration_s:.2f}s")
	return {"rows_indexed": res.rows_indexed, "dataset_id": dataset_id}
