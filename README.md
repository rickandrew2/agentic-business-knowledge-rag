# Agentic Business Knowledge RAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to extract, process, and intelligently query business knowledge from multiple data sources. Built with FastAPI backend, Qdrant vector database, and LLM integration for semantic search and context-aware responses.

## Overview

This RAG system combines cutting-edge generative AI with business data retrieval, enabling intelligent question-answering, document analysis, and knowledge extraction. The system is built with security-first principles, comprehensive testing, and AI-assistant friendly development patterns for easy extension and customization.

## Key Features

- **Multi-Format Data Ingestion**: Process CSV, PDF, Markdown, and other document formats
- **Vector Embeddings**: Semantic search using sentence transformers and embeddings
- **Qdrant Integration**: High-performance vector storage and similarity search
- **FastAPI Backend**: High-performance, async Python API with automatic documentation
- **Security-First**: Input validation, authentication, rate limiting, and secure error handling
- **Comprehensive Testing**: Unit, integration, and E2E test coverage with pytest
- **Scalable Architecture**: Service layer pattern for easy addition of new features
- **LLM Ready**: Designed for integration with GPT, Claude, or open-source models
- **Accessibility & Compliance**: WCAG 2.1 AA standards and enterprise security patterns

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Virtual environment (venv or conda)
- Git

### 2. Clone and Setup

```bash
git clone https://github.com/rickandrew2/agentic-business-knowledge-rag.git
cd agentic-business-knowledge-rag
cd rag-assistant/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the RAG System

```bash
# Start the FastAPI server
python -m uvicorn app.main:app --reload

# Server will be available at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Ingest Sample Data

```bash
# Run sample data ingestion
python -m tests.sample_data

# or use provided test script
powershell -ExecutionPolicy Bypass -File test_upload.ps1
```

## Directory Structure

```
├── rag-assistant/                       # Main RAG system
│   ├── backend/                        # FastAPI backend
│   │   ├── app/
│   │   │   ├── main.py                # FastAPI application entry point
│   │   │   ├── core/
│   │   │   │   ├── config.py          # Configuration and settings
│   │   │   │   └── security.py        # Security utilities and validation
│   │   │   ├── api/
│   │   │   │   ├── schemas.py         # Request/response data models
│   │   │   │   └── routes/            # API endpoints
│   │   │   │       └── data.py        # Data ingestion endpoints
│   │   │   ├── rag/
│   │   │   │   ├── embeddings.py      # Embedding generation
│   │   │   │   ├── ingestion.py       # Document ingestion pipeline
│   │   │   │   └── retriever.py       # Retrieval logic
│   │   │   ├── db/                    # Database initialization
│   │   │   └── log_config/            # Logging configuration
│   │   ├── tests/
│   │   │   ├── unit/                  # Unit tests
│   │   │   ├── integration/           # Integration tests
│   │   │   └── e2e/                   # End-to-end tests
│   │   ├── qdrant_data/               # Qdrant vector storage
│   │   ├── requirements.txt           # Python dependencies
│   │   └── pytest.ini                 # Pytest configuration
│   ├── data/                          # Sample data files
│   │   ├── sample-sales.csv
│   │   └── sample-feedback.md
│   ├── frontend/                      # React frontend (future)
│   └── TESTING_ROADMAP.md             # Testing and implementation roadmap
├── agents/                          # Agent rules and skills
│   ├── rules/                      # Development rules and patterns
│   │   ├── core.mdc               # Core architecture guidelines
│   │   ├── security.mdc           # Security patterns and requirements
│   │   ├── testing.mdc            # Testing strategy and patterns
│   │   ├── frontend.mdc           # Frontend development patterns
│   │   ├── database.mdc           # Database design and patterns
│   │   └── style.mdc              # Code style and formatting rules
│   └── skills/                     # Reusable agent skills
├── .github/
│   └── copilot-instructions.md    # GitHub Copilot configuration
├── AGENTS.md                       # Agent responsibilities and usage guide
├── CLAUDE.md                       # Project guidelines and architecture
├── AI_INSTRUCTIONS.md              # Universal AI assistant instructions
├── RAG_SYSTEM_EXPLAINED.md         # Detailed RAG system documentation
├── .cursorrules                    # Cursor AI assistant configuration
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## Core Configuration Files

### Documentation Files

- **`RAG_SYSTEM_EXPLAINED.md`** - Detailed explanation of the RAG system architecture and components
- **`CLAUDE.md`** - Overall project guidelines, architecture principles, and technology stack selection
- **`AGENTS.md`** - Agent delegation patterns, responsibilities, and when to use each agent
- **`AI_INSTRUCTIONS.md`** - Universal instructions for any AI assistant (reference explicitly)

### AI Assistant Configuration

- **`.cursorrules`** - Cursor-specific rules (auto-loaded by Cursor)
- **`.github/copilot-instructions.md`** - GitHub Copilot instructions (auto-loaded)
- **`agents/rules/*.mdc`** - Detailed rules with `alwaysApply: true` for auto-loading in Cursor

### Rule Files

- **`agents/rules/core.mdc`** - Core architecture principles and best practices
- **`agents/rules/security.mdc`** - Security patterns, OWASP Top 10 protection
- **`agents/rules/testing.mdc`** - Testing strategy, coverage targets, patterns
- **`agents/rules/frontend.mdc`** - Frontend development patterns and accessibility
- **`agents/rules/database.mdc`** - Database schema design and query optimization
- **`agents/rules/style.mdc`** - Code style guidelines and formatting rules

## Agent-Based Development

This RAG system uses specialized agents for different aspects of development:

### BackendAgent (RAG Core)
Manages FastAPI endpoints, vector database integration, embedding generation, and LLM context retrieval.

### DatabaseAgent (ChromaDB)
Owns vector database schema, document ingestion pipelines, collection management, and embedding optimization.

### SecurityAgent (Data Protection)
Ensures secure data handling, input validation for document uploads, authentication for API endpoints, and audit logging.

### TestAgent (Quality Assurance)
Implements comprehensive testing for ingestion pipelines, API endpoints, retrieval accuracy, and end-to-end workflows.

### ReviewerAgent (Code Quality)
Reviews code for correctness, security vulnerabilities, test coverage, performance, and documentation.

## Core Principles

### Security-First Development
- Validate all inputs at application boundaries with schema validation
- Authenticate and authorize every protected endpoint
- Rate limit public endpoints to prevent abuse
- Sanitize outputs to prevent injection attacks
- Never expose sensitive data in error messages or logs

### Testing Strategy
- **Unit Tests**: 80% coverage for business logic
- **Integration Tests**: 15% coverage for API endpoints and database operations
- **E2E Tests**: 5% coverage for critical user journeys
- **Accessibility Tests**: WCAG 2.1 AA compliance for all UI

### Type Safety
- Use strong typing systems available in your chosen language
- Implement runtime validation for all external/user-provided data
- Validate at boundaries: API endpoints, form submissions, configuration
- Generate types from schemas when possible (OpenAPI, GraphQL, database schemas)

### Code Quality
- Maintain consistent patterns throughout the codebase
- Follow established code style guidelines
- Ensure proper error handling and logging
- Keep documentation updated with code changes

## Core Principles

### Security-First Development
- Validate all inputs at application boundaries with Pydantic schemas
- Implement authentication for sensitive endpoints
- Rate limit document ingestion to prevent resource abuse
- Sanitize document data to prevent injection attacks
- Never expose sensitive business data in error messages
- Log security events appropriately with audit trails

### Testing Strategy
- **Unit Tests**: 80% coverage for embeddings and retrieval logic
- **Integration Tests**: 15% coverage for API endpoints and Qdrant operations
- **E2E Tests**: 5% coverage for complete ingestion-to-retrieval workflows
- **Security Tests**: Input validation and authentication flow testing

### Type Safety
- Use Python type hints throughout the codebase
- Implement Pydantic models for all request/response data
- Validate at boundaries: API endpoints, document uploads
- Generate OpenAPI schemas automatically from type annotations

### Code Quality
- Maintain consistent MVC-style architecture
- Follow Python PEP 8 style guidelines
- Ensure comprehensive docstrings and inline comments
- Keep documentation synchronized with API changes

## RAG System Architecture

### Data Ingestion Pipeline
1. **Input**: CSV, PDF, Markdown, or text files
2. **Processing**: Document chunking and cleaning
3. **Embedding**: Convert text to vector embeddings using sentence transformers
4. **Storage**: Store vectors and metadata in ChromaDB
5. **Indexing**: Automatic indexing for fast similarity search

### Query Pipeline
1. **Input**: User question or query
2. **Embedding**: Convert query to vector using same embeddings model
3. **Retrieval**: Find similar vectors in Qdrant (top-k results)
4. **Context**: Package retrieved documents as context
5. **Output**: Return ranked results with relevance scores

### API Endpoints

**Health & Status**
- `GET /` - Health check
- `GET /status` - System status

**Data Ingestion**
- `POST /api/data/ingest` - Upload and process documents
- `GET /api/data/collections` - List ingested collections
- `DELETE /api/data/collections/{collection_id}` - Delete collection

**Query & Retrieval**
- `POST /api/data/query` - Search ingested data
- `GET /api/data/query/{query_id}` - Retrieve query details

## Security Features

- **Input Validation**: Pydantic schemas for all API inputs
- **Rate Limiting**: Request throttling for public endpoints
- **File Upload Security**: Content type validation and size limits
- **Error Handling**: Secure error responses without data leaks
- **Logging**: Audit trails for all data access

## Testing Strategy

**Unit Tests** (`tests/unit/`)
```bash
pytest tests/unit/ -v
```

**Integration Tests** (`tests/integration/`)
```bash
pytest tests/integration/ -v
```

**End-to-End Tests**
```bash
pytest tests/ -v --tb=short
```

**With Coverage Report**
```bash
pytest --cov=app --cov-report=html
```

## AI Assistant Integration

### Cursor
Rules are automatically loaded from:
- `.cursorrules` file
- `agents/rules/*.mdc` files with `alwaysApply: true`
- `AGENTS.md` and `CLAUDE.md` (via workspace rules)

### GitHub Copilot
Instructions are automatically loaded from:
- `.github/copilot-instructions.md`

### Other AI Assistants
Reference `AI_INSTRUCTIONS.md` explicitly in your prompts:
- "Follow the patterns in `AI_INSTRUCTIONS.md`"
- "Check `AGENTS.md` for agent delegation"
- "Apply security patterns from `agents/rules/security.mdc`"

## Development Workflows

### Adding a New Data Source

1. **Create Ingestion Handler** (BackendAgent)
   - Add parser in `app/rag/ingestion.py`
   - Define schema in `app/api/schemas.py`

2. **Add API Endpoint** (BackendAgent)
   - Create route in `app/api/routes/data.py`
   - Implement validation and error handling

3. **Write Tests** (TestAgent)
   - Unit test for parser
   - Integration test for endpoint
   - E2E test for full pipeline

4. **Update Documentation** (ReviewerAgent)
   - Add to API docs
   - Update README or RAG_SYSTEM_EXPLAINED.md

### Improving Retrieval Accuracy

1. **Review Embedding Quality** (DatabaseAgent)
   - Test with different embedding models
   - Evaluate retrieval results

2. **Optimize Chunking** (DatabaseAgent)
   - Adjust chunk size and overlap
   - Implement metadata filtering

3. **Add Hybrid Search** (BackendAgent)
   - Combine vector search with keyword matching
   - Implement ranking strategy

4. **Test and Validate** (TestAgent)
   - Measure retrieval accuracy
   - Benchmark performance

## Documentation

### Getting Started
1. Review `RAG_SYSTEM_EXPLAINED.md` for system architecture
2. Check `TESTING_ROADMAP.md` for testing and implementation status
3. Reference `AGENTS.md` for agent responsibilities and delegation
4. Configure your AI assistant using the appropriate configuration file

### Additional Resources
- **System Architecture**: See `RAG_SYSTEM_EXPLAINED.md` for detailed architecture
- **Agent Usage**: See `AGENTS.md` for detailed agent responsibilities
- **Security**: See `agents/rules/security.mdc` for security patterns
- **Testing**: See `agents/rules/testing.mdc` for testing strategy
- **Code Style**: See `agents/rules/style.mdc` for formatting rules
- **Testing Roadmap**: See `TESTING_ROADMAP.md` for current implementation status

## Quality Gates

All code must meet these standards:
- Pass linting and formatting checks
- Meet minimum test coverage thresholds (80% unit, 15% integration, 5% E2E)
- Pass security scans without high-severity issues
- Implement proper error handling and validation
- Include comprehensive docstrings and comments
- Follow established security patterns

## Contributing

When contributing to this RAG system:
1. Follow enterprise-grade security patterns
2. Update relevant rule files in `agents/rules/` if patterns change
3. Keep documentation synchronized with code changes
4. Ensure comprehensive test coverage for new features
5. Follow Python PEP 8 style guidelines
6. Maintain AI assistant configuration compatibility

## Project Status

Current implementation includes:
- ✅ FastAPI backend with async support
- ✅ QDrant integration for vector storage
- ✅ Document ingestion pipeline (CSV, Markdown, text)
- ✅ Semantic search and retrieval API
- ✅ Security validation and error handling
- ✅ Unit and integration tests
- ✅ API documentation (Swagger/OpenAPI)

Planned enhancements:
- [ ] PDF ingestion with OCR
- [ ] LLM integration for response generation
- [ ] Advanced filtering and hybrid search
- [ ] React frontend dashboard
- [ ] Performance optimization and benchmarking
- [ ] Multi-user authentication system

---

**Ready to extend the RAG system?** Choose your next feature, follow the development workflows, and maintain enterprise-grade patterns throughout.
