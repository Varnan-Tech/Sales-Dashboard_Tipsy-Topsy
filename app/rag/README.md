# RAG Integration (Feature Flagged)

This package integrates a production-oriented RAG pipeline into the existing Streamlit dashboard. It is disabled by default and gated by `ENABLE_RAG=false`.

## Enable

Set in your environment:

```
ENABLE_RAG=true
REDIS_URL=redis://localhost:6379/0
OPENROUTER_API_KEY=sk-...
CHROMA_PERSIST_DIR=.chroma
EMB_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

Then import and mount in your app:

```python
from app.rag.streamlit_integration import mount_rag_ui
if os.getenv("ENABLE_RAG", "false").lower() == "true":
	mount_rag_ui(st)
```

## Local Dev

```
docker compose -f docker-compose.rag.yml up -d redis
```

Run worker:

```
python -m rq worker --url ${REDIS_URL} default
```

## Notes
- All secrets via env vars. Never hardcode.
- Background jobs publish to Redis channels `INDEX_PROGRESS:{dataset_id}`.
