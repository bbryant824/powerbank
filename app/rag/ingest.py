# app/rag/ingest.py
from __future__ import annotations
import os, asyncio, tempfile
from pathlib import Path
from telegram import File as TgFile
from app.rag.pdf_loader import extract_chunks
from app.vector_store.chroma_client import get_user_collection
import logging
log = logging.getLogger(__name__)

def store_pdf_path(pdf_path: Path, chat_id: int, file_name: str) -> int:
    """SYNC: chunk → embed → write. Returns the number of vectors ACTUALLY written."""
    store  = get_user_collection(chat_id)
    before = store._collection.count()
    log.info("[INGEST] chat=%s before=%s path=%s", chat_id, before, pdf_path)

    chunks = extract_chunks(pdf_path)
    log.info("[INGEST] chat=%s chunks_split=%s", chat_id, len(chunks))

    if not any(ch.strip() for ch in chunks):
        log.warning("[INGEST] chat=%s empty/extract_failed", chat_id)
        return 0

    try:
        store.add_texts(chunks, metadatas=[{"source": file_name}] * len(chunks))
    except Exception as e:
        log.exception("[INGEST] chat=%s add_texts failed: %s", chat_id, e)
        return 0

    after  = store._collection.count()
    wrote  = max(0, after - before)
    log.info("[INGEST] chat=%s after=%s wrote=%s", chat_id, after, wrote)
    return wrote

async def store_pdf_async(tg_file: TgFile, chat_id: int, file_name: str) -> int:
    """ASYNC: download (await) → offload heavy work to a thread."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        temp_path = Path(tmp.name)

    try:
        # PTB v22: async — must await
        await tg_file.download_to_drive(custom_path=str(temp_path))
        size = os.path.getsize(temp_path)
        log.info("[INGEST] chat=%s downloaded=%s bytes", chat_id, size)

        wrote = await asyncio.to_thread(store_pdf_path, temp_path, chat_id, file_name)
        return wrote
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
