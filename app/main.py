# app/main.py
from __future__ import annotations
from fastapi import FastAPI, Request
from telegram import Update
from contextlib import asynccontextmanager

from app.bot.handlers import build_application
from app.config import settings

# Build both apps
telegram_app = build_application(settings.telegram_token)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- START PTB ---
    await telegram_app.initialize()
    await telegram_app.start()
    print("PTB started:", telegram_app.running)   # should print True
    try:
        yield
    finally:
        # --- STOP PTB ---
        await telegram_app.stop()
        await telegram_app.shutdown()
        print("PTB stopped")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Welcome to LearnBot"}

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, telegram_app.bot)
    print("Webhook hit:", data.get("update_id"))
    await telegram_app.update_queue.put(update)
    return {"ok": True}
