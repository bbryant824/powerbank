# app/rag/retrieve_tool.py
from __future__ import annotations
from typing import List, Tuple
from langchain_core.tools import tool
from app.vector_store.chroma_client import get_user_collection

@tool
def retrieve(question: str, chat_id: int, k: int = 4) -> Tuple[str, List[str]]:
    """Retrieve up to k relevant chunks from the user's collection (requires chat_id)."""
    store = get_user_collection(chat_id)
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    chunks = [d.page_content for d in docs]
    # short header + raw chunks; ReAct will see this text in its scratchpad
    header = f"Retrieved {len(chunks)} chunks for question: {question}"
    return header, chunks
