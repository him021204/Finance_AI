# Finance AI Assistant

A full-stack RAG-powered financial Q&A chatbot. Upload annual reports (PDF/ZIP), and the AI answers questions with cited sources from the documents.

Built with **FastAPI + FAISS + Sentence Transformers** on the backend and **React + Vite** on the frontend.

---

## Features

- **RAG Pipeline** — Retrieval-Augmented Generation over uploaded annual reports
- **Multi-Report Knowledge Base** — Upload multiple PDFs or ZIP files; all reports are searchable together
- **Persistent Vector Store** — FAISS index saved to disk; fast startup on subsequent runs
- **Auto-Ingestion** — Place PDFs/ZIPs in `server/knowledge_base/` and they're vectorized on first server start
- **Markdown Responses** — Bot answers render with bold, lists, tables, code blocks
- **Chat History** — Past conversations saved in the sidebar
- **ChatGPT-style UI** — Dark theme, typing indicators, suggestion chips

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite, Vanilla CSS, Lucide Icons, react-markdown |
| **Backend** | Python, FastAPI, Uvicorn |
| **Embeddings** | Sentence Transformers (`BAAI/bge-small-en-v1.5`) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **LLM** | Qwen2.5-0.5B-Instruct (swap to `Himanshu2124/qwen-finance-7b` for full model) |
| **PDF Parsing** | pdfplumber (text + table extraction) |

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- **Git**
- ~4 GB disk space (for model downloads on first run)
- GPU optional — runs on CPU (slower) or CUDA GPU (faster)

---

## Setup Guide

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd Finance_AI
```

### Step 2 — Backend Setup

Open a terminal and navigate to the server directory:

```bash
cd server
```

#### 2a. Create a Python virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 2b. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, the embedding model (`BAAI/bge-small-en-v1.5`) and LLM (`Qwen/Qwen2.5-0.5B-Instruct`) will be downloaded automatically from Hugging Face. This may take a few minutes.

#### 2c. (Optional) Pre-load annual reports

Place PDF files or ZIP archives (containing PDFs) in the `server/knowledge_base/` folder:

```
server/
  knowledge_base/
    HDFC_Annual_Report_2024.pdf
    Reliance_Reports.zip        ← ZIP with multiple PDFs inside
```

These will be automatically vectorized when the server starts for the first time. The vector store is saved to `knowledge_base/_vector_store/` so subsequent starts are instant.

#### 2d. Start the backend server

```bash
python server.py
```

The server runs at **http://localhost:8000**. You'll see logs like:

```
Loading embedding model …
Embedding model ready.
Loading LLM: Qwen/Qwen2.5-0.5B-Instruct …
LLM ready.
Knowledge base ready: 3 files, 847 total chunks
```

### Step 3 — Frontend Setup

Open a **second terminal** and navigate to the client directory:

```bash
cd client
```

#### 3a. Install npm dependencies

```bash
npm install
```

#### 3b. Start the development server

```bash
npm run dev
```

The frontend runs at **http://localhost:5173**.

### Step 4 — Open the App

Open your browser and go to:

```
http://localhost:5173
```

You're ready to chat! Upload reports via the sidebar or start asking questions.

---

## Project Structure

```
Finance_AI/
├── client/                     # React frontend
│   ├── src/
│   │   ├── App.jsx             # Main application component
│   │   ├── index.css           # All styles (ChatGPT-style dark theme)
│   │   └── main.jsx            # React entry point
│   ├── package.json
│   └── vite.config.js
│
├── server/                     # Python backend
│   ├── server.py               # FastAPI server with RAG pipeline
│   ├── requirements.txt        # Python dependencies
│   └── knowledge_base/         # Drop PDFs/ZIPs here for auto-ingestion
│       └── _vector_store/      # Persisted FAISS index (auto-generated)
│           ├── index.faiss
│           ├── chunks.pkl
│           └── meta.json
│
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed server status |
| `POST` | `/upload-multiple` | Upload multiple PDF/ZIP files to knowledge base |
| `POST` | `/query` | Ask a question (uses knowledge base or general LLM) |
| `DELETE` | `/session/{id}` | Delete a specific session |

### Upload Example

```bash
curl -X POST http://localhost:8000/upload-multiple \
  -F "files=@report1.pdf" \
  -F "files=@report2.pdf"
```

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "knowledge_base", "question": "What was the revenue?", "history": []}'
```

---

## How It Works

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Embedding   │────▶│  FAISS Index  │────▶│  Top-K Chunks│
│  (bge-small) │     │  (vector DB)  │     │  Retrieved   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │   LLM Prompt  │
                                        │  (Qwen + RAG) │
                                        └──────┬───────┘
                                               │
                                               ▼
                                         Final Answer
                                        (with citations)
```

1. **PDF Parsing** — pdfplumber extracts text and tables from each page
2. **Chunking** — Text split into overlapping 512-char chunks with source metadata
3. **Embedding** — Chunks vectorized using `BAAI/bge-small-en-v1.5`
4. **Indexing** — Vectors stored in a FAISS inner-product index
5. **Retrieval** — User query embedded → top-5 similar chunks retrieved
6. **Generation** — Retrieved context + question sent to LLM for answer with citations

---

## Configuration

Key settings in `server.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | LLM model (change to `Himanshu2124/qwen-finance-7b` for full model) |
| `EMBED_MODEL_ID` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_NEW_TOKENS` | `512` | Max tokens in LLM response |
| `DEVICE` | Auto-detected | `cuda` if GPU available, else `cpu` |

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the server directory |
| Server crashes on model load | Ensure 4+ GB RAM free; try smaller model |
| Frontend can't connect | Ensure backend is running on port 8000 |
| PowerShell blocks `npm run dev` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Slow responses on CPU | Expected — GPU recommended for production use |
| ZIP upload fails | Ensure ZIP contains `.pdf` files (not nested ZIPs) |
