# app/bot/handlers.py
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import asyncio, os, tempfile
from pathlib import Path

from telegram import Update, constants
from telegram.ext import (
    Application, ContextTypes,
    CommandHandler, MessageHandler, filters,
)

from app.vector_store.chroma_client import get_user_collection
from app.rag.pdf_loader import extract_chunks
from app.graph.graph_builder import graph   # imports the compiled LangGraph


# ─── helper: /start ───────────────────────────────────────────
async def start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Send me a PDF, then ask me questions about it. 🤖📚"
    )

# … imports unchanged …
from app.rag.ingest import store_pdf
from app.rag.retrieve_tool import retrieve      # for typing only

async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.mime_type != constants.DocumentMimeType.PDF:
        return await update.message.reply_text("Please send a PDF 🙂")

    user_id = update.effective_chat.id
    tg_file = await ctx.bot.get_file(doc.file_id)

    n_chunks = await asyncio.to_thread(
        store_pdf, tg_file, user_id, doc.file_name
    )

    await update.message.reply_text(
        f"Indexed {n_chunks} chunks from “{doc.file_name}”. Ask away!"
    )


# ─── helper: user question ────────────────────────────────────
async def on_text(update: Update, _: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    question = update.message.text

    state = {
        "chat_id":  user_id,
        "messages": [{"role": "user", "content": question}],
    }
    answer_state = await asyncio.to_thread(graph.invoke, state)
    reply = answer_state["messages"][-1]["content"]
    await update.message.reply_text(reply)

# ─── factory: return a ready PTB Application ──────────────────
def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.PDF, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app
