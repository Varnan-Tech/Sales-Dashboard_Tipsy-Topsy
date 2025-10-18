# RAG Runbook

## Deploy Steps
1. Set `ENABLE_RAG=true` in staging.
2. Provide `REDIS_URL`, `OPENROUTER_API_KEY`, `CHROMA_PERSIST_DIR`.
3. Start Redis and RQ worker.
4. Verify smoke tests and feature UI loads.

## Rollback
- Disable feature: set `ENABLE_RAG=false`.
- Delete indices/collections per dataset via cleanup script (to be extended):
```
python app/rag/scripts/cleanup_indices.py --dataset DATASET_ID
```
- Clear Redis keys with prefix `INDEX_*` for the dataset.

## Load Testing
- 10 concurrent users, 1000 queries, monitor latency and error rates.

## Acceptance Tests
- Upload small CSV (10 rows). Query known value; verify provenance.
- Out-of-context query returns exactly: `I don't know.`
- Re-index does not duplicate vectors (idempotent ids).
- App runs unchanged with `ENABLE_RAG=false`.
