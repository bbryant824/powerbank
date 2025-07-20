# app/vector_store/chroma_client.py
from __future__ import annotations

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import settings

embed_fn = OpenAIEmbeddings(
    model=settings.embed_model,
    openai_api_key=settings.openai_key,
)

def get_user_collection(chat_id: int) -> Chroma:
    tenant   = f"user_{chat_id}" # Unique tenant per user
    database = "learnbot"

    # ── 1️⃣  CONNECT AS ADMIN (no tenant/database) ─────────────────────
    admin_client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )._admin_client   # low-level admin API

    # ── 2️⃣  ENSURE TENANT & DATABASE EXIST ───────────────────────────
    try:
        admin_client.get_tenant(name=tenant)
    except chromadb.errors.NotFoundError:
        admin_client.create_tenant(name=tenant)

    try:
        admin_client.get_database(name=database, tenant=tenant)
    except chromadb.errors.NotFoundError:
        admin_client.create_database(name=database, tenant=tenant)

    # ── 3️⃣  NOW CONNECT AS THAT TENANT ───────────────────────────────
    user_client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        tenant=tenant,
        database=database,
    )

    # LangChain wrapper (autocreates collection if missing)
    return Chroma(
        client=user_client,
        collection_name="docs",
        embedding_function=embed_fn,
    )
