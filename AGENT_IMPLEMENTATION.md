# Agent System Implementation Log

## Overview

Adding an EcoMind-style multi-agent system on top of ResearchMate's existing RAG + recommendation stack.

**Goal:** Replace isolated `/qa/ask` and `/feed/generate` calls with a single conversational `/chat` endpoint backed by specialized agents that automatically pick the right tools.

---

## Architecture

```
User message (POST /api/v1/chat)
        │
        ▼
 IntentRecognizer          ← 3-stage hybrid: keywords → embeddings → LLM
        │
        ├── research_qa ──────────► ResearchAgent     (tools: search_knowledge_base)
        ├── recommendation ───────► RecommendationAgent (tools: get_feed, search_kb)
        ├── document_management ──► DocumentAgent      (tools: list_docs, search_docs)
        └── general_chat ─────────► GeneralAgent       (no tools)
                │
                ▼
       ConversationMemory          ← Redis (TTL 24h) with in-memory fallback
                │
                ▼
       LLM-as-Judge (optional)    ← scores reply on 4 dimensions
                │
                ▼
         ChatResponse
```

---

## New Files

| File | Purpose |
|------|---------|
| `src/agents/__init__.py` | Package export |
| `src/agents/intent_recognizer.py` | Hybrid 3-stage intent classification |
| `src/agents/memory.py` | Redis conversation history |
| `src/agents/tools.py` | Tool schemas + executors |
| `src/agents/base_agent.py` | OpenAI function-calling loop |
| `src/agents/research_agent.py` | Research Q&A specialist |
| `src/agents/recommendation_agent.py` | Feed/paper recommendation specialist |
| `src/agents/document_agent.py` | Document management specialist |
| `src/agents/router.py` | AgentRouter dispatch |
| `src/evaluation/__init__.py` | Package export |
| `src/evaluation/llm_judge.py` | LLM-as-Judge (relevance/accuracy/completeness/usefulness) |
| `src/api/routers/chat.py` | `/chat` + `/eval/run` endpoints |

---

## New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chat` | Main conversational entry point with agent routing |
| `GET` | `/api/v1/chat/history/{session_id}` | Retrieve conversation history |
| `DELETE` | `/api/v1/chat/history/{session_id}` | Clear a conversation session |
| `POST` | `/api/v1/chat/eval/run` | Batch LLM-as-Judge evaluation |

---

## Intent Recognition — 3-Stage Hybrid

**Why hybrid:** Single-stage approaches each have blind spots.
- Keywords alone miss paraphrases ("shed light on X" → research_qa, no keyword hit)
- Embeddings alone are slow to bootstrap and miss domain-specific shortcuts
- LLM alone is expensive and adds 300-500ms latency per request

**Pipeline:**

```
Stage 1: Keyword matching       → confidence ≥ 0.6 → return immediately
Stage 2: Embedding cosine sim   → score ≥ 0.65    → return
Stage 3: LLM classification     → always returns (fallback)
```

**Intent categories:**
- `research_qa` — explain/define/compare concepts, summarize papers
- `recommendation` — suggest papers, get feed, trending topics
- `document_management` — list/search uploaded documents
- `general_chat` — greetings, meta-questions, off-topic

---

## Agent Tool-Calling Loop

Each agent uses OpenAI function calling (structured tool use):

```
1. Build messages: [system_prompt] + history[-8] + [user_message]
2. Call OpenAI with tools=agent.tool_schemas
3. If finish_reason == "tool_calls":
     execute each tool call → append results as tool messages
     repeat (max 5 iterations)
4. If finish_reason == "stop":
     return final text + tools_called + citations
```

**Tools available per agent:**

| Agent | Tools |
|-------|-------|
| ResearchAgent | `search_knowledge_base` |
| RecommendationAgent | `get_personalized_feed`, `search_knowledge_base` |
| DocumentAgent | `list_user_documents`, `search_user_documents` |
| GeneralAgent | (none) |

---

## Conversation Memory

- **Backend:** Redis (`REDIS_URL` env var) with automatic in-process dict fallback
- **Key schema:** `chat:{user_id}:{session_id}`
- **TTL:** 24 hours
- **Window:** Last 20 messages stored; last 8 fed to LLM per turn
- **Session ID:** UUID generated on first message, returned to client for continuity

---

## LLM-as-Judge Evaluation

**Dimensions (each 0–10):**
- `relevance` (30%) — Does the answer address the question?
- `accuracy` (30%) — Is it factually grounded in retrieved sources?
- `completeness` (20%) — Does it cover all aspects?
- `usefulness` (20%) — Is it actionable for a researcher?

**`overall` = weighted average of the four dimensions**

**`POST /api/v1/chat/eval/run` accepts:**
```json
{
  "test_cases": [
    {"question": "explain BERT", "expected_intent": "research_qa"},
    {"question": "suggest RL papers"}
  ]
}
```

**Returns:** per-case scores + `intent_accuracy` + `avg_scores`

---

## Progress

- [x] Project structure explored and understood
- [x] `AGENT_IMPLEMENTATION.md` created
- [x] `src/agents/` module — all 9 files implemented and syntax-checked
- [x] `src/evaluation/llm_judge.py` implemented
- [x] `src/api/routers/chat.py` — `/chat`, `/chat/history`, `/chat/eval/run` endpoints
- [x] `chat_router` wired into `main.py` and `routers/__init__.py`
- [x] `src/agents/metrics.py` — Prometheus Counter/Histogram/Gauge definitions
- [x] `chat.py` instrumented with `record_request()` after every response
- [x] `GET /metrics` endpoint added to `main.py` (Prometheus scrape target)
- [x] `prometheus/prometheus.yml` — scrape config (job: researchmate_api, interval: 15s)
- [x] `docker-compose.yml` — Prometheus service + `prometheus_data` volume
- [x] `prometheus-client>=0.19.0` added to `requirements.txt`
- [ ] Frontend integration (new chat UI panel)
- [ ] End-to-end smoke test with real API keys

---

## Implementation Notes

- `AgentRouter` is a module-level singleton (initialized once on first `/chat` request) — shares the already-loaded sentence-transformer with `EmbeddingManager` to avoid loading it twice
- All tool executors are synchronous; FastAPI runs them in a thread pool via `run_in_executor` (not needed since OpenAI calls are already blocking — kept simple)
- Redis failure is fully transparent: `ConversationMemory` falls back to a class-level dict; the user never sees an error
- `/chat` does not replace `/qa/ask` — both coexist so existing frontend integrations are unaffected
