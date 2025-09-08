# app/bot/handlers.py
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import asyncio, os, tempfile
from pathlib import Path

from telegram import Update

from telegram.ext import (
    Application, ContextTypes,
    CommandHandler, MessageHandler, filters,
)
from langchain_core.messages import HumanMessage

from app.vector_store.chroma_client import get_user_collection
from app.rag.pdf_loader import extract_chunks
from app.graph.graph_builder import graph   # imports the compiled LangGraph
import logging
logger = logging.getLogger(__name__)

# ─── helper: /start ───────────────────────────────────────────
async def start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Send me a PDF, then ask me questions about it. 🤖📚"
    )

# … imports unchanged …
from app.rag.ingest import store_pdf_async
from app.rag.retrieve_tool import retrieve      # for typing only

async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    user_id = update.effective_chat.id
    tg_file = await ctx.bot.get_file(doc.file_id)

    wrote = await store_pdf_async(tg_file, user_id, doc.file_name)

    if wrote <= 0:
        return await update.message.reply_text(
            "I couldn’t index that PDF (maybe it’s scanned or empty). "
            "Try a text-based PDF."
        )

    await update.message.reply_text(
        f"Indexed {wrote} chunks from “{doc.file_name}”. Ask away!"
    )


# ─── helper: user question ────────────────────────────────────
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.agent.react_agent import build_agent_for_chat
from app.agent.history import get_history

async def on_text(update: Update, _: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_chat.id
    question = update.message.text

    agent = build_agent_for_chat(chat_id=user_id)

    agent_with_history = RunnableWithMessageHistory(
        agent,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Seed a system message once per chat (optional but nice)
    hist = get_history(f"tg:{user_id}")
    if not hist.messages:
        hist.add_message(SystemMessage(content=SYSTEM_TEXT))

    cfg = {"configurable": {"session_id": f"tg:{user_id}"}}

    result = await asyncio.to_thread(
        agent_with_history.invoke,
        {"input": question},
        cfg
    )
    reply = result["output"] if isinstance(result, dict) and "output" in result else str(result)
    await update.message.reply_text(reply)

from langchain_core.runnables.history import RunnableWithMessageHistory
from app.agent.react_agent import build_agent_for_chat, SYSTEM_TEXT
from app.agent.history import get_history
from langchain_core.messages import SystemMessage
    
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Update caused error", exc_info=context.error)
    try:
        if hasattr(update, "message") and update.message:
            await update.message.reply_text("Oops! I hit an error. Please try again.")
    except Exception:
        pass

# ─── factory: return a ready PTB Application ──────────────────
def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.PDF, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error) 
    return app
