# 0. Activate venv (if not already)
source .venv/bin/activate

# 1. Start (or resume) Chroma in Docker
docker start chroma 2>/dev/null || docker run -d --name chroma \
  -p 8000:8000 \
  -v $(pwd)/chroma_data/:/chroma/.chroma/index \
  ghcr.io/chroma-core/chroma:1.0.15

# 2. Export environment variables (or rely on .env)
export OPENAI_API_KEY=<your-openai-key>
export TELEGRAM_BOT_TOKEN=<botfather-token>

# 3. Launch FastAPI + PTB (port 8002)
uvicorn app.main:app --reload --port 8002

# 4. Expose locally via ngrok (dev only)
ngrok http 8002  # copy HTTPS URL from console

# 5. Tell Telegram where to POST updates
curl -F "url=https://<ngrok-id>.ngrok-free.app/webhook" \
     "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook"

# 6. Verify connectivity (should return JSON about your bot)
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"


2 Key Components & Flow
Stage	File/function	What it does
Webhook in	main.py /webhook	Receives Telegram POST, converts to Update, puts it on PTB queue.
Dispatch	bot.handlers.build_application()	PTB routes to appropriate async handler.
PDF upload	on_document → rag.ingest.store_pdf()	Downloads file, extract_chunks(), embeds with OpenAI, stores vectors in Chroma tenant user_<chat_id>.
Question	on_text	Builds state, calls graph.invoke(state).
LangGraph	graph_builder.py	Node decide (few-shot) → maybe retrieve tool call → generate final answer.
Similarity search	rag.retrieve_tool.retrieve()	Uses get_user_collection(chat_id) → as_retriever().invoke(question) (cosine).
Vector helper	vector_store.chroma_client.get_user_collection()	Creates tenant & DB on first use; returns LangChain Chroma wrapper with embeddings bound.