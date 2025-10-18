from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np

SYSTEM_PROMPT = (
	"You are a sales data analyst. Each document represents one sales transaction with details like date, product, brand, size, quantity, and value. "
	"Analyze patterns across multiple transactions to answer questions about sales performance, popular products, brands, sizes, etc. "
	"ONLY use the transaction data provided in the context. Look for patterns and aggregate information from multiple records. "
	"If the question asks for 'top selling', 'most popular', or 'best performing', analyze quantities and values across all provided transactions. "
	"When you see negative quantities, those are returns/refunds. Focus on positive sales transactions for sales analysis. "
	"Include specific examples and totals from the data. If you cannot find relevant information in the provided transactions, say \"I don't know.\" "
	"Do not make assumptions or use external knowledge."
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
	# Try multiple query variations to improve matching
	query_variations = [
		query,  # Original query
		query.replace("what's", "what is").replace("it's", "what is"),  # Normalize contractions
		query.split("and")[0].strip() if "and" in query else query,  # Try first part
		"sales transactions day 1",  # Specific terms
		"top selling brand",
		"most popular size color"
	]

	all_hits = []
	for q in query_variations[:3]:  # Try first 3 variations
		try:
			q_vec = embeddings.embed_batch([q])[0]
			hits = vector_store.query(dataset_id, np.asarray(q_vec), top_k=top_k//2)  # Fewer per query
			all_hits.extend(hits)
		except:
			continue

	# Remove duplicates and sort by score
	seen_ids = set()
	unique_hits = []
	for hit in sorted(all_hits, key=lambda x: x.get('score', 0), reverse=True):
		hit_id = hit.get('id', '')
		if hit_id not in seen_ids:
			unique_hits.append(hit)
			seen_ids.add(hit_id)
		if len(unique_hits) >= top_k:
			break

	hits = unique_hits
	context, provenance = assemble_context(hits, k=8)

	# Debug: Show what we found
	debug_info = f"\n\nDEBUG INFO: Found {len(hits)} documents. Context length: {len(context)} chars. First hit score: {hits[0].get('score', 'N/A') if hits else 'N/A'}"

	prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
	start = time.time()
	try:
		text = call_openrouter(SYSTEM_PROMPT, prompt)
		# Debug: If LLM returns "I don't know", it might be because context is empty or irrelevant
		if text.strip().lower() == "i don't know." and context.strip():
			text = f"I found some potentially relevant data but couldn't extract a definitive answer. Here's what I found:\n\n{context[:500]}...{debug_info}"
		else:
			text = text + debug_info
	except Exception as e:
		text = f"Error processing query: {str(e)}. Context length: {len(context)}{debug_info}"
	return {
		"answer_text": text,
		"provenance": provenance,
		"latency_s": time.time() - start,
	}
