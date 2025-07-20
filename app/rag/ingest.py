from __future__ import annotations
import os, tempfile
from pathlib import Path
from typing import Tuple

from telegram import File as TgFile
from app.rag.pdf_loader import extract_chunks
from app.vector_store.chroma_client import get_user_collection

def store_pdf(tg_file: TgFile, chat_id: int, file_name: str) -> int:
    """
    Download *tg_file*, chunk it, embed & store vectors in Chroma
    tenant corresponding to *chat_id*.  Returns number of chunks stored.
    """
    # 1. download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tg_file.download_to_drive(tmp.name)
        pdf_path = Path(tmp.name)

    # 2. chunk
    chunks = extract_chunks(pdf_path)
    os.unlink(pdf_path)

    # 3. embed + store
    store = get_user_collection(chat_id)
    metas = [{"source": file_name}] * len(chunks)
    store.add_texts(chunks, metadatas=metas)

    return len(chunks)
