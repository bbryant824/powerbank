Architecture
text
Copy
Edit
Telegram app ─► Telegram servers ─► FastAPI /webhook ─► PTB queue
                                                        │
                                                        ▼
               ┌────────────── app/rag/ingest.py ──────────────┐
/start ─────► greeting                                   │
PDF ────────► download → chunk → embed → Chroma (Docker) │
Question ───► LangGraph (decide → tools → generate) ────►│
               └───────────────────────────────────────────────┘
FastAPI: public HTTPS endpoint, forwards every update to python-telegram-bot (PTB).

Chroma 1.x (Docker): vector DB, tenant = user_<chat_id>.

LangGraph: decides whether to retrieve; uses retrieve tool for similarity search.

MemorySaver: checkpoints conversation state each turn.

Project Layout
text
Copy
Edit
app/
├ main.py                FastAPI entry + /webhook
├ bot/
│   └ handlers.py        /start, PDF upload, Q&A
├ rag/
│   ├ pdf_loader.py      PDF → text chunks
│   ├ ingest.py          download → chunk → embed → store
│   └ retrieve_tool.py   @tool retrieve(question, chat_id)
├ vector_store/
│   └ chroma_client.py   get_user_collection(chat_id)
└ graph/
    └ graph_builder.py   LangGraph: decide → tools → generate
Tech Stack & Versions
Layer	Library	Version
LLM & Chains	langchain	0.3.26
Core & Tools	langchain-core, langchain-community	0.3.70 / 0.0.52
LangGraph	langgraph	0.5.3
Vector DB	chromadb (server 1.0.15 in Docker)	1.0.15
Web & Bot	fastapi / python-telegram-bot	0.116.1 / 22.2
Embeddings	openai	1.97.0
Settings	pydantic / pydantic-settings	2.7.4 / 2.1.0
Tracing (opt)	langsmith	latest

Environment Variables
dotenv
Copy
Edit
# .env
OPENAI_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=8130732892:...
CHROMA_HOST=localhost
CHROMA_PORT=8000
EMBED_MODEL=text-embedding-3-small

# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=LearnBot-dev
Dotenv loading:
app/config.py runs

python
Copy
Edit
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
Boot-Up Cheat-Sheet
bash
Copy
Edit
# 1  Activate venv
source .venv/bin/activate

# 2  Start (or run) Chroma
docker start chroma 2>/dev/null || docker run -d --name chroma \
  -p 8000:8000 \
  -v $(pwd)/chroma_data/:/chroma/.chroma/index \
  ghcr.io/chroma-core/chroma:1.0.15

# 3  Launch FastAPI + PTB
uvicorn app.main:app --reload --port 8002

# 4  Dev tunnel (ngrok)
ngrok http 8002                       # copy HTTPS URL

# 5  Register webhook
curl -F "url=https://<ngrok-id>.ngrok-free.app/webhook" \
     "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook"

# 6  Verify
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
End-to-End Runtime Flow
6.1 PDF Upload
Webhook POST → FastAPI → PTB on_document.

rag.ingest.store_pdf()

download file → temp

extract_chunks() (pypdf + splitter)

get_user_collection(chat_id) creates tenant user_<id> if missing

add_texts() embeds via OpenAI and stores vectors in Chroma.

Bot replies “Indexed N chunks from ‘file.pdf’”.

6.2 User Question
on_text builds state and calls graph.invoke(state).

LangGraph nodes

decide – LLM with few-shots; may emit retrieve tool-call.

ToolNode – executes rag.retrieve_tool.retrieve() (similarity search in user tenant).

generate – builds final prompt (context + question) and calls GPT-4o-mini.

Bot sends answer.

6.3 Conversation Memory
MemorySaver checkpoints state dict after each node, so multi-turn context persists automatically.

Implementation Highlights
Feature	Where	Notes
Tenant isolation	chroma_client.py	tenant = user_<chat_id>, DB = learnbot
Few-shot tool selection	graph_builder.py	generic Q → direct answer, study Q → retrieval
Dockerised Chroma	docker run …	vectors persisted in ./chroma_data
LangSmith tracing	.env flags	every graph run recorded in UI

Testing & Debugging
Task	Command
Count vectors for user	store = get_user_collection(id); store._collection.count()
Peek first doc	store._collection.peek(1)['documents'][0][:200]
Chroma logs	docker logs -f chroma
FastAPI requests	uvicorn console (--log-level debug)
Inspect inbound webhook	http://127.0.0.1:4040 (ngrok)
LangSmith runs	smith.langchain.com → LearnBot-dev

Next Milestones
Containerise FastAPI app – Dockerfile + docker-compose.yml (app + chroma).

Pytest suite – unit test chunker; integration test upload-and-ask.

CI/CD – GitHub Actions building images, pushing to ECR, running tests.

Duplicate upload guard – hash PDFs before embedding.

Rate limiting & quotas – protect cost and abuse.

Production deploy – AWS App Runner or ECS Fargate with stable HTTPS domain.

6.1 PDF Upload
Telegram → FastAPI
POST /webhook with Update (document).

FastAPI → PTB queue
Update enqueued.

handlers.on_document

downloads file to temp

rag.ingest.store_pdf()

extract_chunks() (pypdf + splitter)

get_user_collection(chat_id) → ensures tenant user_<id>

add_texts() → embeddings computed & stored in Chroma

replies “Indexed N chunks…”

6.2 User Question
handlers.on_text builds state = {"chat_id", "messages":[…]}.

LangGraph (graph_builder.py)

decide node (few-shot)

generic Q → skip retrieval

study Q → emits retrieve tool-call

ToolNode executes retrieve_tool.retrieve() (similarity search).

generate node builds final prompt (context if any) → GPT-4o-mini.

Answer sent back via PTB bot.

6.3 Memory
MemorySaver checkpoints state after each node; history auto-loads on next turn.