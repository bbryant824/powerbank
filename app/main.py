from fastapi import FastAPI, Request
from telegram import Update
from app.bot.handlers import build_application        # your handlers builder
from app.config import settings                       # loads TELEGRAM_BOT_TOKEN

# 1  create both frameworks
app          = FastAPI()
fastapi_app  = app            # alias for uvicorn
telegram_app = build_application(settings.telegram_token)
print("DEBUG OPENAI key loaded? ", bool(settings.openai_key))
print("DEBUG TELEGRAM token loaded? ", bool(settings.telegram_token))

# 2  expose root for quick health-check
@app.get("/")
async def root():
    return {"message": "Welcome to LearnBot"}

# 3  Telegram webhook endpoint
@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, telegram_app.bot)
    print("Webhook hit:", data["update_id"])
    await telegram_app.update_queue.put(update)
    return {"ok": True}
