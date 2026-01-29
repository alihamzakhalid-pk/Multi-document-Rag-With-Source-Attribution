# 📚 Multi-Document RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers questions using uploaded documents with **strict source attribution** and **zero-hallucination** guarantees.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **Zero Hallucination** - Answers grounded only in retrieved documents
- 📍 **Exact Page Numbers** - DOCX files auto-convert to PDF for accurate page tracking
- 📄 **Multi-Format Support** - PDF, DOCX, and TXT files
- 🔗 **Source Attribution** - Every answer includes document + page references
- ⚡ **Fast API** - Built with FastAPI + ChromaDB vector store
- 🧠 **Groq LLM** - Uses Llama 3.1 70B for answer generation

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/alihamzakhalid-pk/Multi-document-Rag-With-Source-Attribution.git
cd Multi-document-Rag-With-Source-Attribution

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Configure API Key

1. Copy `.env.example` to `.env`
2. Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 💡 Get a free API key at [console.groq.com](https://console.groq.com)

### Step 3: Run the Server

```powershell
python main.py
```

🎉 **Server running at:** http://localhost:8000

---

## 📖 Usage

### Web Interface
Open http://localhost:8000 in your browser for the UI.

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/documents/upload` | Upload a document |
| `GET` | `/api/v1/documents` | List all documents |
| `DELETE` | `/api/v1/documents/{name}` | Delete a document |
| `POST` | `/api/v1/query` | Ask a question |

### Example: Upload & Query

```powershell
# Upload a document
curl -X POST "http://localhost:8000/api/v1/documents/upload" -F "file=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/api/v1/query" `
  -H "Content-Type: application/json" `
  -d '{"question": "What is the main topic?"}'
```

### Response Format

```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "document_name": "document.pdf",
      "page": 3,
      "chunk_id": "document.pdf_p3_c2_abc123"
    }
  ]
}
```

---

## 📁 Project Structure

```
Multi/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── .env                        # Your API key (create from .env.example)
│
├── api/
│   ├── routes.py               # API endpoints
│   └── schemas.py              # Request/response models
│
├── rag/
│   ├── document_loader.py      # Load PDF, DOCX, TXT
│   ├── document_converter.py   # DOCX → PDF conversion
│   ├── chunker.py              # Text chunking
│   ├── embeddings.py           # Vector embeddings
│   ├── vector_store.py         # ChromaDB storage
│   ├── retriever.py            # Similarity search
│   └── answer_generator.py     # LLM answer generation
│
├── config/
│   └── settings.py             # Configuration
│
├── static/
│   └── index.html              # Web UI
│
└── tests/                      # Unit tests
```

---

## ⚙️ Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *required* | Your Groq API key |
| `LLM_MODEL` | `llama-3.1-70b-versatile` | LLM model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks to retrieve |

---

## 📝 Requirements

- **Python 3.9+**
- **Groq API Key** (free tier available)
- **Microsoft Word** (optional, for DOCX → PDF conversion)

---

## 🧪 Run Tests

```powershell
python -m pytest tests/ -v
```

---

## 📄 License

MIT License
