# ResearchMate

ResearchMate is a multi-user research learning platform with a personalized paper feed, routed research agents, retrieval over papers and user documents, and a Coding Agent for turning papers into teaching implementations or exploring public repositories.

**Live:** https://www.researchmate.site

## Features

- **Personalized feed** — indexes arXiv papers, retrieves candidates from ChromaDB, and ranks them with an 11-feature heuristic or per-user LightGBM LambdaRank model.
- **Agent chat** — routes research questions, recommendations, document requests, and general conversation. Research agents can use the knowledge base, full-paper retrieval, and Tavily web search when configured.
- **Deep research** — plans up to five subtasks, gathers evidence, and streams a cited report.
- **Personal knowledge base** — accepts PDF, TXT, DOCX, and Markdown uploads with user-scoped retrieval.
- **Coding Agent** — uses LangGraph for paper implementation and public repository exploration.
- **Evaluation and observability** — provides deterministic tests, offline and paid model evaluations, tool traces, and Prometheus metrics.

## Agent routing

The main chat router classifies requests as research, recommendation, document management, or general chat. Research questions are then dispatched by query shape.

| Request | Agent or flow |
|---|---|
| Simple research question, streaming | `ResearchAgent` |
| Simple research question, non-streaming | `ReflectionAgent` |
| Explicit comparison or trade-off | `PlanAndSolveAgent` |
| Long or multi-part research request | `DeepResearchAgent` |
| Feed or paper recommendation | `RecommendationAgent` |
| User document request | `DocumentAgent` |
| Greeting, meta question, or off-topic request | `GeneralAgent` in `router.py` |
| Paper implementation or repository exploration | LangGraph Coding Agent |

The supervisor is retained as an offline experiment and is not part of the production router.

## Coding Agent

The paper workflow checks whether available evidence supports a single-file teaching implementation, generates annotated Python, optionally executes and repairs it, and reports runtime success separately from evidence faithfulness.

The repository workflow pins a public GitHub repository to a commit, inspects selected files, explains architecture, answers grounded follow-up questions, and can compare teaching code with a candidate repository module.

Coding Agent jobs use a durable database queue:

```text
Browser → FastAPI queues PracticeRun in PostgreSQL
        → practice-worker claims the job
        → Paper or Repo LangGraph writes durable checkpoints
        → browser polls the owner-scoped run endpoint
```

The worker-related names have distinct responsibilities:

| Component | Trust boundary | Responsibility |
|---|---|---|
| `src/jobs/practice_worker.py` | Trusted application process | Runs model calls, advances LangGraph, resumes checkpoints, and writes results. It never executes generated Python. |
| `sandbox_worker/` | Isolated untrusted-code service | Executes generated or edited Python with network, filesystem, CPU, memory, and timeout restrictions. |
| `src/agents/coding_runtime.py` | Shared library | Owns the process-wide checkpointer and graph run/resume helpers. |

Paper draft generation and repository exploration work without the sandbox. Code execution and automatic repair require a separately deployed sandbox configured through `PRACTICE_WORKER_URL` and `PRACTICE_WORKER_TOKEN`.

## Technology

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, SQLAlchemy |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Agent orchestration | OpenAI tool loop; LangGraph for Coding Agent workflows |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Persistent ChromaDB |
| Ranking | LightGBM LambdaRank with 11 served features |
| Database | PostgreSQL in production, SQLite for development/tests |
| Cache and traces | Redis with documented in-process fallbacks |
| Deployment | Docker Compose, Nginx, GitHub Actions |

## Local development

Prerequisites: Python 3.11+, Node.js 18+, and an OpenAI API key for model-backed features.

```bash
git clone https://github.com/SamLv0114/Two_Mode_Learning_Assistant.git researchmate
cd researchmate
cp .env.example .env

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Start the trusted Coding Agent orchestration worker in a second terminal:

```bash
source venv/bin/activate
python -m src.jobs.practice_worker
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at http://localhost:3000 and API documentation at http://localhost:8000/docs.

For the production-shaped stack:

```bash
cp .env.example .env
docker compose up -d --build
```

The core Compose stack has seven services: PostgreSQL, Redis, API, trusted Coding Agent worker, frontend, Prometheus, and Nginx. The isolated code-execution worker deliberately runs outside this stack on a separate host.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Chat, summarization, evaluation, and Coding Agent generation |
| `SECRET_KEY` | Stable JWT signing key; required outside tests |
| `DATABASE_URL` | SQLite default or production PostgreSQL connection |
| `REDIS_URL` | Conversation memory, fact memory, traces, and caches |
| `TAVILY_API_KEY` | Enables the `search_web` tool |
| `SEMANTIC_SCHOLAR_API_KEY` | Raises ingestion and citation-refresh API capacity |
| `PRACTICE_ENABLED` | Enables Coding Agent API/UI capabilities |
| `PRACTICE_WORKER_URL` | HTTPS endpoint of the external isolated sandbox |
| `PRACTICE_WORKER_TOKEN` | Authentication token shared with the sandbox |
| `PRACTICE_TASK_POLL_SECONDS` | Trusted task worker database poll interval |

See `.env.example` for the full set and optional feature flags.

## Main API groups

| Prefix | Responsibility |
|---|---|
| `/api/v1/auth` | Registration, login, token refresh, profile |
| `/api/v1/feed` | Generate, poll, list, and refine feeds |
| `/api/v1/interactions` | Feedback and ranking model status/retraining |
| `/api/v1/chat` | Routed chat, SSE, history, trace, evaluation |
| `/api/v1/qa` | User document upload, listing, deletion |
| `/api/v1/whats_hot` | Trending research content |
| `/api/v1/practice` | Coding Agent runs, discussion, comparison, execution |

## Repository layout

```text
src/
  agents/
    coding_practice.py       # Paper and Repo LangGraph definitions
    coding_runtime.py        # shared checkpoint/run/resume helpers
    coding_services.py       # evidence, model, GitHub, sandbox adapters
    deep_research_agent.py
    router.py                # routing and GeneralAgent
    tools.py
  api/routers/
    chat.py
    feed.py
    practice.py
    qa.py
    whats_hot.py
  collectors/
    paper_data.py            # shared PaperData record
    arxiv_rss_collector.py
  jobs/
    practice_worker.py       # trusted Coding Agent task process
    nightly_indexer.py
    bulk_import.py
  models/
    feature_extractor.py
    user_recommender.py
    user_trainer.py
frontend/src/components/
  PracticeWorkspace.tsx
  PracticeRunViews.tsx
evaluation/
  datasets/
  runners/
  results/
sandbox_worker/              # separately deployed untrusted-code executor
```

## Evaluation

Deterministic tests do not call paid model APIs:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke.py
```

Registered evaluation tracks:

```bash
python evaluation/run_all.py --help
```

Paid model evaluations are explicitly invoked. The Coding Agent runner requires a curated JSONL case file and writes a dated output unless `CODING_AGENT_EVAL_OUTPUT` is supplied:

```bash
CODE_LAB_EVAL_CASES=evaluation/datasets/my_cases.jsonl \
python evaluation/runners/run_code_lab_eval.py
```

`RUN:PASS` means a synthetic program check ran successfully; it does not reproduce the paper's experimental claims. `FAITHFUL:PASS` is a constrained model review over cited evidence, not independent human verification.
