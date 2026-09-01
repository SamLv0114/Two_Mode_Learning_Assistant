# ResearchMate

A multi-user ML research platform — personalized paper and article recommendations, multi-agent chat with RAG, and a personal knowledge base that learns from your interactions.

**Live:** https://www.researchmate.site

---

## Features

- **Daily Feed** — ArXiv papers fetched nightly via RSS and ranked per-user with a LightGBM learning-to-rank model (13 features). Tech articles discovered daily via Tavily web search. Two modes: *Recommended* (ChromaDB semantic retrieval + LTR) and *Latest* (pure recency).
- **Multi-Agent Chat** — SSE streaming chat with automatic intent routing to five specialized agents. Intent classified in three stages: keyword rules → embedding cosine similarity → GPT-4o-mini.
- **Paper Deep Dive** — "Ask Agent" button on any paper card injects the paper's title and abstract as context on the first message so agents can give grounded, specific answers.
- **Personal Knowledge Base** — Upload PDF, TXT, DOCX, or Markdown files. Chunks are embedded into ChromaDB and become searchable by all agents in future conversations.
- **User Fact Memory** — GPT-4o-mini extracts facts about you after each turn (interests, expertise, goals) and stores them in Redis for 30 days. These are prepended to agent system prompts to personalize every response.
- **Adaptive Ranking** — Per-user LightGBM model auto-trains after 50 interactions (saved+viewed+dismissed). Falls back to heuristic scoring before enough data exists.
- **LLM-as-Judge Evaluation** — Batch eval endpoint scores agent replies on groundedness, completeness, and clarity using CriticAgent (gpt-4o-mini). Also used inline by ReflectionAgent for quality control.
- **Prometheus Metrics** — Request counts, latency histograms, tool usage, and eval scores exposed at `/metrics`.

---

## Agent routing

All `research_qa` messages go through intent recognition; the router then picks a sub-agent based on query shape.

| Query type | Signal | Agent |
|---|---|---|
| Comparison / trade-off | compare, vs, pros and cons, when to use… | PlanAndSolveAgent |
| Multi-faceted or long | >90 chars, or survey / comprehensive / multiple… | DeepResearchAgent |
| Simple (non-streaming) | everything else, non-streaming path | ReflectionAgent |
| Simple (streaming) | everything else, SSE path | ResearchAgent |
| Feed / recommendation | recommend, trending, show me papers… | RecommendationAgent |
| Document management | my files, knowledge base, uploaded… | DocumentAgent |
| General | greetings, meta questions, off-topic | GeneralAgent |

**DeepResearchAgent** runs a three-stage pipeline: `TodoPlanner` (gpt-4o) decomposes the question into up to 5 sub-tasks → `TaskSummarizer` (gpt-4o-mini) searches ChromaDB and the web per task → `ReportWriter` (gpt-4o) synthesizes a structured Markdown report, streamed token by token. Progress events (`plan`, `task_started`, `task_done`) arrive over SSE before the first token.

**ReflectionAgent** wraps ResearchAgent with a one-shot critique loop. CriticAgent scores groundedness, completeness, and clarity (0–1 each). If the aggregate falls below 0.70, the critique is injected back and ResearchAgent produces a refined answer.

**PlanAndSolveAgent** generates a numbered analysis blueprint, executes each step with a focused KB search, then synthesizes all step results into a final Markdown answer.

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| API | FastAPI · Uvicorn · APScheduler | REST + SSE, nightly indexer at 06:00 UTC |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS | Dashboard, feed, streaming chat |
| LLM | OpenAI API (gpt-4o / gpt-4o-mini) | All agent reasoning, summarization, evaluation |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors for ChromaDB and intent classification |
| Vector DB | ChromaDB (embedded, persistent) | Paper and document semantic search |
| Ranking | LightGBM LambdaRank · scikit-learn | Per-user learning-to-rank (13 features) |
| Database | PostgreSQL (prod) · SQLite (dev) | Users, papers, articles, interactions, model state |
| Cache | Redis *(optional)* | Session memory, fact memory, feed job status, tool traces |
| Web search | Tavily *(optional)* | Article discovery, agent `search_web` tool |
| Auth | JWT (HS256) · bcrypt | 24 h access tokens, password hashing |
| Observability | Prometheus · `/metrics` | Request counts, latency, tool calls, eval scores |
| Deployment | Docker Compose · Nginx | Six-service stack (db, redis, api, frontend, prometheus, nginx) |

---

## Quickstart (local dev)

**Prerequisites:** Python 3.11+, Node 18+

### 1. Clone and configure

```bash
git clone https://github.com/your-username/researchmate
cd researchmate
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and SECRET_KEY
```

### 2. Start the backend

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Interactive API docs are at http://localhost:8000/docs.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:3000
```

### 4. Seed the paper index

The nightly indexer runs automatically at 06:00 UTC. To populate papers immediately:

```bash
curl -X POST http://localhost:8000/api/v1/admin/index-now
```

### Docker Compose (production)

```bash
cp .env.example .env            # fill in all values
docker-compose up -d
```

This starts PostgreSQL, Redis, the API, Next.js, Prometheus, and Nginx. The frontend is served at `:80`. Prometheus runs at `:9090`.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used by all agents, fact extraction, and evaluation. |
| `SECRET_KEY` | Yes | JWT signing key. Generate once: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Changing it invalidates all sessions. |
| `DATABASE_URL` | — | Defaults to `sqlite:///./data/learning_assistant.db`. Use `postgresql://user:pw@host/db` for production. |
| `REDIS_URL` | — | Enables session memory, fact memory, feed job tracking, and tool-call traces. Falls back to in-memory without it. |
| `TAVILY_API_KEY` | — | Enables `WebArticleAgent` (daily article discovery) and the `search_web` agent tool. Articles are skipped without it. |
| `LLM_MODEL` | — | Default model for ResearchAgent and GeneralAgent. Default: `gpt-4o-mini`. DeepResearch and PlanSolve always use gpt-4o for planning. |
| `ARXIV_CATEGORIES` | — | ArXiv categories to index. Default: `cs.LG,cs.AI,cs.CV,cs.CL,cs.NE`. |
| `CORS_ORIGINS` | — | Allowed origins, comma-separated. Default: `http://localhost:3000,http://localhost:8000`. |

---

## API reference

All `/api/v1/` routes except `/auth/register` and `/auth/login` require `Authorization: Bearer <token>`. Full docs at `/docs` (Swagger UI).

### Auth
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create account. Returns user + access token. |
| `POST` | `/api/v1/auth/login` | Form login. Returns JWT access token. |
| `GET` | `/api/v1/auth/me` | Current user profile. |
| `PUT` | `/api/v1/auth/me` | Update interests, focus areas, or password. |

### Feed
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/feed/generate` | Start background feed generation job. Returns `job_id`. |
| `GET` | `/api/v1/feed/status/{job_id}` | Poll job status (`pending` → `running` → `completed`). |
| `GET` | `/api/v1/feed/papers` | Paginated recommended papers for current user. |
| `GET` | `/api/v1/feed/articles` | Paginated recommended articles. |
| `GET` | `/api/v1/feed/saved` | All saved papers and articles. |

### Chat (agent)
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat/stream` | SSE streaming. Yields `intent`, `agent`, `plan`, `task_*`, `token`, `done` events. |
| `POST` | `/api/v1/chat/` | Non-streaming chat. Set `enable_eval: true` for LLM-as-Judge scores. |
| `GET` | `/api/v1/chat/history/{session_id}` | Retrieve conversation history. |
| `DELETE` | `/api/v1/chat/history/{session_id}` | Clear a session. |
| `GET` | `/api/v1/chat/trace/{session_id}` | Tool-call trace (tool, args, result preview, latency). Expires after 1 hour. |
| `POST` | `/api/v1/chat/eval/run` | Batch eval over test cases. Returns per-case scores and averages. |

### Knowledge base
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/qa/documents` | Upload a document (PDF, TXT, MD). Chunks and embeds into ChromaDB. |
| `GET` | `/api/v1/qa/documents` | List uploaded documents for current user. |
| `DELETE` | `/api/v1/qa/documents/{id}` | Delete document from both the database and ChromaDB. |

### Interactions
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/interactions` | Record viewed / saved / dismissed on a paper or article. |
| `GET` | `/api/v1/interactions/stats` | Interaction counts and model training status. |
| `POST` | `/api/v1/interactions/model/retrain` | Manually trigger per-user model retraining. |

### Observability
| Method | Path | Description |
|---|---|---|
| `GET` | `/metrics` | Prometheus metrics: `agent_requests_total`, `agent_latency_seconds`, `tool_calls_total`, `agent_eval_score`. |
| `POST` | `/api/v1/admin/index-now` | Manually run the nightly ArXiv indexer. |

---

## Project structure

```
researchmate/
├── src/
│   ├── agents/
│   │   ├── base_agent.py           # OpenAI tool-calling loop + listener hook
│   │   ├── router.py               # AgentRouter — intent → agent dispatch
│   │   ├── intent_recognizer.py    # keyword → embedding → LLM (3-stage)
│   │   ├── research_agent.py       # RAG Q&A with ChromaDB + web search
│   │   ├── reflection_agent.py     # ResearchAgent + CriticAgent critique loop
│   │   ├── deep_research_agent.py  # TodoPlanner → TaskSummarizer → ReportWriter
│   │   ├── plan_and_solve_agent.py # structured Plan → Solve → Synthesize
│   │   ├── recommendation_agent.py # fetches + presents personalized feed
│   │   ├── document_agent.py       # KB list and search
│   │   ├── critic_agent.py         # LLM-as-judge (groundedness/completeness/clarity)
│   │   ├── memory.py               # ConversationMemory + UserFactMemory (Redis)
│   │   ├── context_builder.py      # semantic history selection via cosine similarity
│   │   ├── tool_aware_agent.py     # wraps any agent, records tool-call traces
│   │   ├── tools.py                # OpenAI function schemas + execute_tool()
│   │   └── web_article_agent.py    # Tavily-powered article discovery
│   ├── api/
│   │   ├── main.py                 # FastAPI app, lifespan, CORS, APScheduler
│   │   └── routers/
│   │       ├── auth.py             # register, login, me, update
│   │       ├── feed.py             # generate, status, papers, articles, saved
│   │       ├── interactions.py     # create, stats, model retrain
│   │       ├── chat.py             # stream, non-stream, history, trace, eval
│   │       └── qa.py               # document upload, list, delete (KB panel)
│   ├── pipelines/
│   │   └── daily_feed.py           # end-to-end feed generation pipeline
│   ├── models/
│   │   ├── embeddings.py           # EmbeddingManager (singleton, thread-safe)
│   │   ├── user_recommender.py     # per-user LightGBM LTR (loaded from DB)
│   │   ├── user_trainer.py         # trains / retrains per-user model
│   │   └── feature_extractor.py    # 13-feature vector builder
│   ├── collectors/
│   │   ├── arxiv_rss_collector.py  # RSS feed parser for ArXiv categories
│   │   └── arxiv_collector.py      # API-based fallback collector
│   ├── jobs/
│   │   ├── nightly_indexer.py      # embed + upsert new papers into ChromaDB
│   │   └── bulk_import.py          # one-shot bulk paper import utility
│   ├── evaluation/
│   │   └── llm_judge.py            # LLMJudge wrapping CriticAgent (0–10 scale)
│   ├── rag/
│   │   ├── retriever.py            # vector search via EmbeddingManager
│   │   └── generator.py            # LLM answer generation with citations
│   ├── database/
│   │   └── models.py               # SQLAlchemy models + init_db()
│   └── utils/
│       ├── config.py               # pydantic-settings with field validators
│       └── preprocessing.py        # PDF (PyMuPDF), HTML, text chunking
└── frontend/
    └── src/
        ├── app/
        │   ├── page.tsx            # landing page
        │   ├── login/page.tsx      # login
        │   ├── register/page.tsx   # registration
        │   └── dashboard/page.tsx  # main app — feed + agent chat
        └── lib/
            └── api.ts              # typed axios API client
```

---

## Architecture overview

```
Ingestion:   ArXiv RSS → NightlyIndexer (06:00 UTC) → PostgreSQL → ChromaDB
Articles:    WebArticleAgent (Tavily) → PostgreSQL → LightGBM LTR

Request:     HTTP / SSE → IntentRecognizer → AgentRouter → Agent → Stream
Agents:      Research · DeepResearch · PlanSolve · Reflection · Document · Rec · General
Memory:      ContextBuilder (cosine sim) + UserFactMemory (Redis) + ConversationMemory
Eval:        CriticAgent (inline) · LLMJudge (batch) · Prometheus metrics
```

---

## License

MIT
