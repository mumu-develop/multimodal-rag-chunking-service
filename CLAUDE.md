# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal RAG Chunking Service — a FastAPI backend that parses documents (PDF/DOCX/PPTX/images), performs multimodal chunking (text/tables/images), embeds content via BGE-M3 + OpenCLIP, and stores results in MySQL + Milvus. Uses Celery + Redis for async processing.

## Directory Structure

```
backend/
├── app/
│   ├── main.py          # FastAPI entry, routes, middleware, trace_id
│   ├── config.py        # Pydantic Settings (.env loading)
│   ├── schemas.py       # Pydantic request/response models
│   ├── models.py        # SQLAlchemy ORM (DocChunk, TaskStatus)
│   ├── chunker.py       # Multimodal chunking (text/table/image)
│   ├── embedder.py      # Embedding wrappers (BGE-M3 + OpenCLIP)
│   ├── storage.py       # MySQL & Milvus read/write operations
│   └── tasks.py         # Celery async pipeline
├── alembic/             # Database migrations
├── requirements.txt
├── Dockerfile
└── .env.example
docker-compose.yml       # Full stack: MySQL, Milvus, Redis, Backend, Celery
tests/e2e_test.sh        # E2E curl test script
```

## Key Commands

```bash
# Local development (requires MySQL, Milvus, Redis running)
cd backend
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# Run Celery worker
cd backend
celery -A app.tasks worker --loglevel=info

# Run full stack with Docker Compose
docker compose up -d

# Run migrations
cd backend
alembic upgrade head

# Run E2E tests
bash tests/e2e_test.sh [path_to_pdf]
```

## Architecture

### Async Pipeline (Celery task chain in `tasks.py`)
`upload → parse → chunk → embed → store_mysql → store_milvus → update_status`

- Progress is reported via `self.update_state(state="PROGRESS", ...)`
- Tasks are persisted in MySQL `tasks` table
- Retry: max 3 attempts with exponential backoff

### Storage
- **MySQL** (`doc_chunks`): Metadata table with chunk_id, doc_id, chunk_type, content, caption, page_num, JSON metadata
- **Milvus** (`rag_chunks`): Vector collection with `text_vector` (1024d, HNSW) and `image_vector` (512d, IVF_FLAT)
- Write order: MySQL first → then Milvus. Either failure marks task as failed.

### Embedding
- Text: BGE-M3 (1024 dim, normalized)
- Image: OpenCLIP ViT-B-32 (512 dim, normalized)
- `EMBED_MOCK=true` in `.env` uses random vectors for dev/testing

### API Contract
All responses wrapped in: `{"success": bool, "code": str, "message": str|null, "trace_id": str, "data": any}`

| Method | Path | Description |
|:---|:---|:---|
| POST | `/api/v1/upload` | Upload file, returns task_id + doc_id |
| GET | `/api/v1/tasks/{task_id}` | Task status + progress |
| GET | `/api/v1/chunks` | List chunks by doc_id (paginated, filterable by type) |
| POST | `/api/v1/search` | Vector search with query + top_k |

## Important Notes

- The `.env` file must exist before starting. Copy `.env.example` to `.env` and adjust values.
- `DATABASE_URI` uses MySQL (`mysql+pymysql://`), not PostgreSQL despite the env variable name in the original blueprint.
- Celery worker runs as a separate service in docker-compose; both backend and worker need the same environment.
- File upload limit: 50MB. Allowed extensions: pdf, docx, pptx, png, jpg, jpeg, md.
- Chunking rules: text uses RecursiveCharacterTextSplitter (800/150), tables >800 chars split by row keeping header, images get VLM caption (mock by default) + 200 char context.
