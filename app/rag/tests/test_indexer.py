import pandas as pd
from app.rag.indexer import iter_documents


def test_iter_documents_basic():
	df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
	docs = list(iter_documents("ds1", df, "file.csv", primary_key=None))
	assert len(docs) == 2
	doc_id, text, meta = docs[0]
	assert doc_id.startswith("ds1-")
	assert meta["file_name"] == "file.csv"
