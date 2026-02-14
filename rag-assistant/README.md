# RAG Business Analytics Assistant

An AI-powered assistant that answers questions from company data (sales, products, customer feedback) using Retrieval-Augmented Generation (RAG).

**Example Queries:**
- "What were our top 3 performing products last quarter?"
- "Summarize customer feedback trends."
- "Which region had the highest growth?"

## Tech Stack

- **Backend**: FastAPI + Python
- **Vector DB**: Chroma (MVP) → Pinecone (scale)
- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI text-embedding-3-small
- **Frontend**: React + TypeScript (coming soon)
- **Database**: SQLite (MVP) → PostgreSQL (production)

## Project Structure

```
rag-assistant/
├── backend/
│   ├── app/
│   │   ├── core/          # Configuration, security
│   │   ├── rag/           # RAG pipeline (ingestion, retrieval, prompting)
│   │   ├── api/           # FastAPI routes
│   │   ├── db/            # Database models
│   │   ├── eval/          # Evaluation metrics
│   │   ├── logging/       # Structured logging
│   │   └── main.py        # FastAPI entry point
│   ├── tests/             # Unit & integration tests
│   └── requirements.txt
├── frontend/              # React app (coming soon)
├── data/                  # Sample data
└── .env.example           # Configuration template
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 2. Setup

```bash
# Clone or access the project
cd rag-assistant/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Server

```bash
# From backend directory
python -m uvicorn app.main:app --reload

# Open http://localhost:8000/docs for API documentation
# Test health endpoint: http://localhost:8000/api/health
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# Get config
curl http://localhost:8000/api/config
```

## Development Roadmap

### MVP (This Week)
- ✅ FastAPI server setup
- ✅ Configuration & security
- [ ] Data ingestion (CSV + markdown)
- [ ] Embedding + Chroma integration
- [ ] Basic retrieval API
- [ ] Chat API with RAG
- [ ] Evaluation metrics
- [ ] React frontend

### V1 (Week 2)
- Chunking optimization (semantic vs fixed-size)
- Retrieval ranking (BM25 + re-ranking)
- Hallucination detection
- Error handling & graceful degradation
- Unit + integration tests (>80% coverage)
- Security hardening (PII detection, prompt injection defense)
- Docker + cloud deployment

### V2 (Week 3+)
- Agentic retrieval (multi-step reasoning)
- Query decomposition
- Custom embeddings or fine-tuning
- Admin dashboard
- Advanced monitoring

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/unit/test_chunking.py -v
```

## API Endpoints

### Health & Config
- `GET /api/health` - Health check
- `GET /api/config` - Public configuration

### Data Management (Coming Soon)
- `POST /api/data/upload` - Upload CSV/documents
- `GET /api/data/status` - Check ingestion progress

### Chat & RAG (Coming Soon)
- `POST /api/chat/message` - Ask question and get answer
- `GET /api/chat/history` - View chat history

### Evaluation (Coming Soon)
- `GET /api/eval/metrics` - Retrieval quality metrics

## Security

### Implemented
- ✅ Input validation with injection prevention
- ✅ PII detection (Email, Phone, SSN, Credit Cards, IP)
- ✅ Rate limiting framework
- ✅ Secure error responses
- ✅ Structured logging

### Coming Soon
- Prompt injection defense refinement
- RBAC authentication
- Audit trail database
- Compliance logging

## Environment Variables

See [.env.example](.env.example) for all available settings:
- `OPENAI_API_KEY` - Required: Your OpenAI API key
- `OPENAI_MODEL` - Default: gpt-4o
- `CHROMA_PATH` - Vector DB path
- `DATABASE_URL` - SQL database connection
- RAG parameters: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, etc.

## Next Steps

1. **Copy .env.example to .env** and add your OpenAI API key
2. **Run the server**: `python -m uvicorn app.main:app --reload`
3. **Test health**: `curl http://localhost:8000/api/health`
4. **Continue to data ingestion** (next milestone)

## Contributing

This is a portfolio project showcasing RAG + fullstack development.

## License

MIT
