# System Architecture

## Overview

ResearchMate is a personalized research feed and AI agent chat system that combines:
- **Multi-agent LLM system** with intent-routed specialized agents
- **RAG (Retrieval-Augmented Generation)** with ChromaDB vector search
- **LightGBM LambdaRank** for personalized feed ranking
- **Real-time SSE streaming** for agent chat visualization

## System Components

### 1. Data Collection Layer

| Collector | Source | Method |
|-----------|--------|--------|
| ArxivCollector | ArXiv API | Time-slice bucketing across 7-day window (intentional: avoids recency bias) |
| HNCollector | Hacker News Firebase API | Title + HN comments as content; **3s HTTP timeout** per item |
| MediumCollector | Medium RSS (`/feed/tag/machine-learning`) | **RSS summary extraction** via BeautifulSoup — no per-URL scraping |
| DevToCollector | Dev.to RSS (`/feed`) | **RSS summary extraction** via BeautifulSoup — no per-URL scraping |

**Performance note (2026-07):** Medium and Dev.to switched from scraping each article URL to parsing the RSS `<summary>` field directly. This eliminates ~60 blocking HTTP requests per feed generation run. HN item timeout also reduced from 10s to 3s.

### 2. Storage Layer
- **PostgreSQL**: Papers, articles, user interactions, auth
- **ChromaDB Vector Database**: Embeddings for semantic search (cosine similarity)
- **Redis**: Conversation memory (TTL 24h); falls back to in-process dict if unavailable

### 3. Processing Layer
- **EmbeddingManager**: Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- **FeatureExtractor**: Extracts ranking features (similarity, recency, citations, upvotes, novelty)
- **Recommender**: LightGBM LambdaRank model trained on user interaction logs (NDCG@10)
- **NoveltyScorer**: Embedding-based novelty; compares against last `NOVELTY_MAX_ITEMS=10` seen items (reduced from 50 for speed)

### 4. LLM Layer
- **Generator**: OpenAI `gpt-4o-mini` for summarization and Q&A
- **Parallel summarization**: `ThreadPoolExecutor(max_workers=4)` runs LLM summary calls concurrently — cuts summarization wall-clock time by up to 4×

### 5. Agent System

```
User message (POST /api/v1/chat)
        │
        ▼
 IntentRecognizer          ← 3-stage hybrid: keywords → embeddings → LLM fallback
        │
        ├── research_qa ──────────► ResearchAgent     (tools: search_knowledge_base)
        ├── recommendation ───────► RecommendationAgent (tools: get_feed, search_kb)
        ├── document_management ──► DocumentAgent      (tools: list_docs, search_docs)
        └── general_chat ─────────► GeneralAgent       (no tools)
```

See `AGENT_IMPLEMENTATION.md` for full agent architecture details.

### 6. Application Modes

#### Mode 1: Daily Feed Pipeline
```
Collect → Filter → Rank → Summarize (parallel) → Store → Deliver
```

1. **Collect**: Fetch papers (ArXiv time-slice) + articles (HN/Medium/DevTo RSS)
2. **Filter**: Remove low-similarity content (threshold 0.3)
3. **Rank**: LightGBM LambdaRank scores items by relevance + novelty
4. **Summarize**: Parallel LLM summaries (ThreadPoolExecutor, max 4 workers)
5. **Store**: Save to PostgreSQL + ChromaDB
6. **Deliver**: Return ranked feed via background job

Background job flow: `POST /feed/generate` → returns `job_id` → client polls `/feed/status/{job_id}` every 2.5s → on `done`, fetches `/feed/papers` + `/feed/articles`

#### Mode 2: Agent Chat (SSE streaming)
```
Message → Intent → Agent → Tool calls → Streamed response
```

1. **Intent Recognition**: 3-stage hybrid (keyword → embedding → LLM)
2. **Agent Dispatch**: Router selects specialized agent
3. **Tool Execution**: RAG search, feed retrieval, document lookup
4. **SSE Streaming**: Token-by-token response streamed to frontend via `fetch` + `ReadableStream`
5. **Memory**: Redis-backed conversation history (last 8 turns fed to LLM)

## Data Flow

### Daily Feed Flow
```
ArXiv (time-sliced) ──┐
HN (top stories)      ├──► Filter ──► LightGBM Rank ──► Parallel LLM Summarize ──► PostgreSQL + ChromaDB
Medium (RSS summary)  │
Dev.to (RSS summary) ─┘
```

### Agent Chat Flow
```
User message ──► IntentRecognizer ──► AgentRouter ──► Agent + Tools ──► SSE stream ──► Frontend
                                                              │
                                                    ConversationMemory (Redis)
```

## Key Technologies

| Category | Technology |
|----------|-----------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Ranking | LightGBM LambdaRank (NDCG@10) |
| LLM | OpenAI `gpt-4o-mini` |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Cache/Memory | Redis |
| Backend | FastAPI (async) |
| Frontend | Next.js 14 (App Router) |
| Infrastructure | Docker Compose (6 containers), Nginx (SSL), Oracle Cloud |
| Observability | Prometheus + custom metrics (intent counts, latency, feed durations) |

## Configuration (key settings)

| Setting | Value | Notes |
|---------|-------|-------|
| `NOVELTY_MAX_ITEMS` | 10 | Reduced from 50; limits embedding comparisons for novelty scoring |
| `MAX_PAPERS_PER_DAY` | 50 | ArXiv fetch limit |
| `TOP_PAPERS_COUNT` | 5 | Papers returned in feed |
| `TOP_ARTICLES_COUNT` | 3 | Articles returned in feed |
| `LLM_MODEL` | `gpt-4o-mini` | Summarization and agent Q&A |
| `NOVELTY_LOOKBACK_DAYS` | 14 | Days to look back for novelty comparison |
