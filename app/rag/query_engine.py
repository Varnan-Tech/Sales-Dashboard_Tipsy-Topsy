from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np

SYSTEM_PROMPT = (
	"You are a sales data analyst assistant. Analyze the provided sales transaction data and answer questions accurately. "
	"ONLY use the context provided below to answer the user. If the answer cannot be inferred from the context, reply exactly: \"I don't know.\" "
	"When providing numbers, include specific details like dates, amounts, products, and customer information from the data. "
	"Include provenance information (file name + row reference) for any factual claims. "
	"Focus on sales metrics, trends, customer behavior, product performance, and business insights. "
	"Do not use external knowledge or make assumptions beyond what's in the provided data."
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o")


def assemble_context(hits: List[Dict], k: int = 8) -> Tuple[str, List[Dict]]:
	selected = hits[:k]
	ctx_lines = []
	for i, h in enumerate(selected):
		m = h.get("metadata", {})
		content = h.get("content", "")  # Get the actual document content
		score = h.get("score", 0.0)
		line = f"--- DOCUMENT {i+1} (Relevance Score: {score:.3f}) ---\n"
		line += f"Source: {m.get('file_name','?')} | Row: {m.get('row_index','?')}\n"
		line += f"Data: {content}\n"
		line += f"Reference: {m.get('file_name','?')}:row_{m.get('row_index','?')}\n"
		ctx_lines.append(line)
	return "\n\n".join(ctx_lines), selected


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
		"max_tokens": 1500,
	}
	r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=timeout_s)
	r.raise_for_status()
	j = r.json()
	return j["choices"][0]["message"]["content"]


def answer_query(vector_store, embeddings, dataset_id: str, query: str, top_k: int = 10) -> Dict:
	q_vec = embeddings.embed_batch([query])[0]
	hits = vector_store.query(dataset_id, np.asarray(q_vec), top_k=top_k)
	context, provenance = assemble_context(hits, k=8)
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
