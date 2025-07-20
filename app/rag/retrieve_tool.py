from __future__ import annotations
from typing import List
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from app.vector_store.chroma_client import get_user_collection

@tool(response_format="content_and_artifact")
def retrieve(question: str, chat_id: int, k: int = 4) -> List[str]:
    """
    Retrieve up to *k* relevant chunks from the user's documents.
    """
    store: Chroma = get_user_collection(chat_id)
    docs = store.as_retriever(search_kwargs={"k": k}).invoke(question)
    return [d.page_content for d in docs]
