from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np

SYSTEM_PROMPT = (
	"You are a data assistant. ONLY use the context below to answer the user. "
	"If the answer cannot be inferred from the context, reply exactly: \"I don't know.\" "
	"Include provenance lines for any factual claims (file name + row_index/row_range). Do not use world knowledge."
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")


def assemble_context(hits: List[Dict], k: int = 3) -> Tuple[str, List[Dict]]:
	selected = hits[:k]
	ctx_lines = []
	for h in selected:
		m = h.get("metadata", {})
		line = f"[{m.get('file_name','?')}:{m.get('row_index','?')}] {m}"  # keep small for now
		ctx_lines.append(line)
	return "\n".join(ctx_lines), selected


def call_openrouter(system_prompt: str, user_prompt: str, timeout_s: int = 30) -> str:
	import requests

	headers = {
		"Authorization": f"Bearer {OPENROUTER_API_KEY}",
		"Content-Type": "application/json",
	}
	payload = {
		"model": OPENROUTER_MODEL,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		"temperature": 0.1,
		"max_tokens": 600,
	}
	r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=timeout_s)
	r.raise_for_status()
	j = r.json()
	return j["choices"][0]["message"]["content"]


def answer_query(vector_store, embeddings, dataset_id: str, query: str, top_k: int = 8) -> Dict:
	q_vec = embeddings.embed_batch([query])[0]
	hits = vector_store.query(dataset_id, np.asarray(q_vec), top_k=top_k)
	context, provenance = assemble_context(hits, k=3)
	prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:" 
	start = time.time()
	try:
		text = call_openrouter(SYSTEM_PROMPT, prompt)
	except Exception:
		text = "I don't know."
	return {
		"answer_text": text,
		"provenance": provenance,
		"latency_s": time.time() - start,
	}
