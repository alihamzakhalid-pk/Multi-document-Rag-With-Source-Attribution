"""
Multi-Document RAG System with Source Attribution

A production-grade Retrieval-Augmented Generation (RAG) system that
answers questions using ONLY retrieved document chunks with strict
source attribution and zero-hallucination guarantees.

Usage:
    uvicorn main:app --reload

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.settings import get_settings
from utils.logger import get_logger
from api.routes import router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Handles startup and shutdown events for the application.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Multi-Document RAG System")
    logger.info("=" * 60)
    
    settings = get_settings()
    
    # Log configuration (without sensitive data)
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"Chunk Size: {settings.chunk_size}, Overlap: {settings.chunk_overlap}")
    logger.info(f"Top-K Results: {settings.top_k_results}")
    logger.info(f"ChromaDB Path: {settings.chroma_persist_dir}")
    
    # Check API key
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        logger.warning("⚠️  GROQ_API_KEY not configured. Set it in .env file.")
    else:
        logger.info("✓ GROQ_API_KEY configured")
    
    logger.info("=" * 60)
    logger.info("RAG System ready to accept requests")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Multi-Document RAG System")


# Create FastAPI application
app = FastAPI(
    title="Multi-Document RAG System",
    description="""
## Production-Grade RAG with Source Attribution

This API provides a retrieval-augmented generation system that:

- **Answers questions using ONLY uploaded documents**
- **Provides explicit source attribution for every answer**
- **Refuses to answer if information is not in the documents**
- **Supports PDF, DOCX, and TXT file formats**

### Core Principles

1. **Zero Hallucination**: All answers are grounded in retrieved context
2. **Source Attribution**: Every factual statement maps to a source
3. **Multi-Document Support**: Combine information from multiple documents

### Workflow

1. Upload documents via `/documents/upload`
2. Query the system via `/query`
3. Receive answers with source citations
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", tags=["Root"])
async def root():
    """Serve the frontend UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "Multi-Document RAG System",
        "version": "1.0.0",
        "description": "Production-grade RAG with source attribution",
        "documentation": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
