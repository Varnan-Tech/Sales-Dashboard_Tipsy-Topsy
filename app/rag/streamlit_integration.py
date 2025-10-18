import os
from typing import Optional

import streamlit as st

ENABLE_RAG = os.getenv("ENABLE_RAG", "false").lower() == "true"


def mount_rag_ui(st_mod: Optional[object] = None) -> None:
	"""Mount RAG UI into existing Streamlit app when ENABLE_RAG is true.

	Safe to import when disabled. Does not modify existing pages by default.
	"""
	if not ENABLE_RAG:
		return

	st_ = st_mod or st
	st_.markdown("## 🤖 RAG Assistant (Beta)")
	st_.info("This feature is gated by ENABLE_RAG and disabled by default.")

	# Placeholder layout: uploader, dataset selector, and chat input
	uploaded = st_.file_uploader("Upload CSV/XLSX for indexing", type=["csv", "xlsx"])  # noqa: F841
	dataset_id = st_.text_input("Dataset ID", value="default")  # noqa: F841
	st_.text_input("Ask a question about your data", key="rag_query")
	if st_.button("Ask"):
		st_.warning("RAG backend not initialized yet. This is a scaffold.")
