# System Architecture

## Overview

The Two-Mode AI Learning Assistant is a comprehensive system that combines:
- **Classical ML** for ranking and filtering
- **Vector Search** for semantic retrieval
- **LLM** for summarization and Q&A generation

## System Components

### 1. Data Collection Layer
- **ArxivCollector**: Fetches papers from ArXiv API
- **HNCollector**: Fetches articles from Hacker News API
- **MediumCollector**: Fetches articles from Medium RSS
- **DevToCollector**: Fetches articles from Dev.to RSS

### 2. Storage Layer
- **SQLite Database**: Stores papers, articles, and user interactions
- **ChromaDB Vector Database**: Stores embeddings for semantic search

### 3. Processing Layer
- **EmbeddingManager**: Generates embeddings using sentence-transformers
- **FeatureExtractor**: Extracts features for ranking (similarity, recency, citations, etc.)
- **Recommender**: Uses Gradient Boosting to rank content by relevance

### 4. LLM Layer
- **Generator**: Unified interface for OpenAI and Anthropic APIs
- Handles summarization and Q&A generation

### 5. Application Modes

#### Mode 1: Daily Feed Pipeline
```
Collect → Filter → Rank → Summarize → Store → Deliver
```

1. **Collect**: Fetch new papers and articles
2. **Filter**: Remove low-similarity content
3. **Rank**: Use ML model to score and rank items
4. **Summarize**: Generate personalized summaries with LLM
5. **Store**: Save to database and vector DB
6. **Deliver**: Format and output results

#### Mode 2: Q&A Assistant
```
Question → Vector Search → Context Retrieval → LLM Generation → Answer + Citations
```

1. **Question**: User asks a question
2. **Vector Search**: Find relevant documents using embeddings
3. **Context Retrieval**: Get top N most relevant documents
4. **LLM Generation**: Generate answer with citations
5. **Output**: Return formatted answer with source citations

## Data Flow

### Daily Feed Flow
```
ArXiv/Articles → Database → Filtering → Feature Extraction → Ranking → 
LLM Summarization → Database Update → Vector DB Update → Output
```

### Q&A Flow
```
User Question → Embedding → Vector Search → Context Documents → 
LLM Prompt → Answer Generation → Formatting → Output
```

## Key Technologies

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (persistent, cosine similarity)
- **Ranking**: Gradient Boosting Regressor (scikit-learn)
- **LLM**: use free api/model first
- **Database**: SQLite (SQLAlchemy ORM)

## Configuration

- User interests (for personalization)
- Number of recommendations
- Similarity thresholds
- LLM provider and model
- Data sources
- API keys

## Future Enhancements
- web app

