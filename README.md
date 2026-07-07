# ResearchMate — Personal Knowledge & Learning Assistant

A multi-user ML-powered platform that recommends research papers and tech articles, answers questions over a personal knowledge base, and learns from user feedback.

**Live:** https://www.researchmate.site

## Features

### Multi-Agent Chat
- Single `/chat` endpoint routes messages to the right specialized agent automatically
- **Intent recognition** uses a 3-stage hybrid: keyword rules → embedding cosine similarity → LLM classification
- **ResearchAgent** answers ML/AI questions grounded in the knowledge base via RAG
- **RecommendationAgent** fetches and presents personalized paper/article suggestions
- **DocumentAgent** searches and manages uploaded personal documents
- Conversation memory persisted in Redis (24h TTL) with automatic in-memory fallback
- **LLM-as-Judge** evaluation scores responses on relevance, accuracy, completeness, and usefulness

### Daily Feed
- Collects papers from ArXiv and articles from HackerNews, Dev.to, and Medium
- Ranks content using LightGBM LambdaRank (learning-to-rank) with 13 engineered features
- Falls back to heuristic scoring with learned weights for new users (cold start)
- Applies epsilon-greedy exploration and MMR diversity re-ranking to prevent filter bubbles
- Generates personalized summaries via OpenAI API

### Q&A Assistant
- Upload PDFs, text, or markdown files to build a personal knowledge base
- Retrieves relevant context using sentence-transformer embeddings and ChromaDB vector search
- Generates cited answers using retrieval-augmented generation (RAG) with OpenAI

### Per-User Personalization
- Each user gets an independent recommendation model trained on their interactions
- Model retrains automatically after 50+ interactions (save/view/dismiss)
- Configurable focus areas (ML, NLP, CV, AI, DL) and custom research interests

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Database | PostgreSQL, Redis, ChromaDB |
| ML | LightGBM, scikit-learn, sentence-transformers |
| LLM | OpenAI API (GPT-4o-mini) |
| Observability | Prometheus |
| Deployment | Docker Compose, Nginx, Oracle Cloud |

## Project Structure

```
src/
  agents/       # Multi-agent system (intent recognizer, router, tools, memory)
  api/          # FastAPI routers (auth, feed, interactions, Q&A, chat)
  collectors/   # ArXiv, HackerNews, Dev.to, Medium scrapers
  database/     # SQLAlchemy models
  evaluation/   # LLM-as-Judge scoring
  models/       # LightGBM ranker, feature extraction, embeddings
  pipelines/    # Feed generation pipeline
  rag/          # Retriever + Generator for Q&A
  utils/        # Config, preprocessing
frontend/       # Next.js app
prometheus/     # Prometheus scrape config
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat` | Conversational chat with agent routing |
| GET | `/api/v1/chat/history/{session_id}` | Retrieve conversation history |
| POST | `/api/v1/chat/eval/run` | Batch LLM-as-Judge evaluation |
| POST | `/api/v1/feed/generate` | Generate personalized feed |
| POST | `/api/v1/qa/ask` | Direct RAG Q&A |
| POST | `/api/v1/qa/documents` | Upload document to knowledge base |
| GET | `/metrics` | Prometheus metrics |

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials
2. Run with Docker Compose:

```bash
docker compose up -d --build
```

The app will be available at `http://localhost:3000` with the API at `http://localhost:8000`.
Prometheus is available at `http://localhost:9090`.
