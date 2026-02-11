# ResearchMate -- Personal Knowledge & Learning Assistant

A multi-user ML-powered platform that recommends research papers and tech articles, answers questions over a personal knowledge base, and learns from user feedback.

**Live:** https://www.researchmate.site

## Features

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
| LLM | OpenAI API |
| Deployment | Docker Compose |

## Project Structure

```
src/
  api/          # FastAPI routers (auth, feed, interactions, Q&A)
  collectors/   # ArXiv, HackerNews, Dev.to, Medium scrapers
  database/     # SQLAlchemy models
  models/       # LightGBM ranker, feature extraction, embeddings
  pipelines/    # Feed generation pipeline
  rag/          # Retriever + Generator for Q&A
  schemas/      # Pydantic request/response schemas
  utils/        # Config, preprocessing
frontend/       # Next.js app
```

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials
2. Run with Docker Compose:

```bash
docker compose up -d --build
```

The app will be available at `http://localhost:3000` with the API at `http://localhost:8000`.
