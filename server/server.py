"""
Financial AI Backend — FastAPI + RAG Pipeline
Model: Himanshu2124/qwen-finance-7b (HuggingFace)
Run locally or on any GPU machine (Kaggle, Colab, RunPod, etc.)

Usage:
    pip install -r requirements.txt
    python server.py
"""

import os
import gc
import re
import json
import uuid
import pickle
import logging
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List

import torch
import uvicorn
import pdfplumber
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ─────────────────────────────
#MODEL_ID        = "Himanshu2124/qwen-finance-7b"
MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID  = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K           = 5
MAX_NEW_TOKENS  = 512
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR      = Path(tempfile.mkdtemp())

# ── Knowledge base config ────────────────────────────────────────
KNOWLEDGE_BASE_DIR   = Path(__file__).parent / "knowledge_base"
KNOWLEDGE_SESSION_ID = "knowledge_base"
VECTOR_STORE_DIR     = KNOWLEDGE_BASE_DIR / "_vector_store"
FAISS_INDEX_FILE     = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_FILE          = VECTOR_STORE_DIR / "chunks.pkl"
META_FILE            = VECTOR_STORE_DIR / "meta.json"

# ── ngrok (optional) ─────────────────────────────────────────────
NGROK_TOKEN = os.getenv("NGROK_TOKEN", "")

log.info(f"Using device: {DEVICE}")

# ─────────────────────────── App ────────────────────────────────
app = FastAPI(
    title="Financial AI API",
    description="RAG-powered Q&A over Annual Reports using Qwen-Finance-7B",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Global state ───────────────────────
sessions: dict[str, dict] = {}
embedder:  SentenceTransformer = None
tokenizer: AutoTokenizer        = None
model:     AutoModelForCausalLM = None

# Track knowledge base files: [{filename, num_chunks, source}]
knowledge_files: list[dict] = []


# ─────────────────────────── Pydantic schemas ───────────────────
class QueryRequest(BaseModel):
    session_id: str
    question:   str
    history:    Optional[list[dict]] = []

class QueryResponse(BaseModel):
    answer:     str
    sources:    list[str]
    session_id: str

class UploadResponse(BaseModel):
    session_id:  str
    num_chunks:  int
    message:     str

class MultiUploadResponse(BaseModel):
    session_id:      str
    total_chunks:    int
    files_processed: list[str]
    files_failed:    list[str]
    message:         str


# ─────────────────────────── Startup: load models ───────────────
@app.on_event("startup")
async def load_models():
    global embedder, tokenizer, model

    log.info("Loading embedding model …")
    embedder = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
    log.info("Embedding model ready.")

    log.info(f"Loading LLM: {MODEL_ID} …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if DEVICE == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    log.info("LLM ready.")

    # ── Auto-ingest knowledge base folder ────────────────────────
    ingest_knowledge_base()


# ─────────────────────────── Vector store persistence ───────────
def save_vector_store():
    """Save FAISS index, chunks and metadata to disk."""
    if KNOWLEDGE_SESSION_ID not in sessions:
        return
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    session = sessions[KNOWLEDGE_SESSION_ID]

    # Save FAISS index
    faiss.write_index(session["index"], str(FAISS_INDEX_FILE))

    # Save chunks
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(session["chunks"], f)

    # Save metadata (file list)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_files, f, indent=2)

    log.info(f"Vector store saved to {VECTOR_STORE_DIR}")


def load_vector_store() -> bool:
    """Try to load persisted vector store. Returns True if successful."""
    global knowledge_files

    if not (FAISS_INDEX_FILE.exists() and CHUNKS_FILE.exists()):
        return False

    try:
        index = faiss.read_index(str(FAISS_INDEX_FILE))

        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)

        sessions[KNOWLEDGE_SESSION_ID] = {
            "index": index,
            "chunks": chunks,
            "filename": "Knowledge Base",
        }

        if META_FILE.exists():
            with open(META_FILE, "r", encoding="utf-8") as f:
                knowledge_files = json.load(f)

        log.info(f"Loaded vector store: {len(chunks)} chunks, {len(knowledge_files)} files")
        return True
    except Exception as e:
        log.error(f"Failed to load vector store: {e}")
        return False


# ─────────────────────────── Knowledge base ingestion ───────────
def ingest_knowledge_base():
    """
    On startup:
    1. If a saved vector store exists, load it directly (fast).
    2. Otherwise, scan for PDFs/ZIPs, vectorize, and save the store.
    """
    global knowledge_files

    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Try loading existing vector store first
    if load_vector_store():
        log.info("Using persisted vector store — skipping re-ingestion.")
        return

    # No saved store — scan and ingest
    pdf_paths = []

    # Collect direct PDFs
    for f in KNOWLEDGE_BASE_DIR.iterdir():
        if f.suffix.lower() == ".pdf":
            pdf_paths.append(f)

    # Extract and collect PDFs from ZIP files
    for f in KNOWLEDGE_BASE_DIR.iterdir():
        if f.suffix.lower() == ".zip":
            log.info(f"Extracting ZIP: {f.name}")
            try:
                extract_dir = KNOWLEDGE_BASE_DIR / f"_extracted_{f.stem}"
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(f, "r") as zf:
                    zf.extractall(extract_dir)
                for pdf in extract_dir.rglob("*.pdf"):
                    pdf_paths.append(pdf)
                log.info(f"Extracted {f.name} — found PDFs inside")
            except Exception as e:
                log.error(f"Failed to extract {f.name}: {e}")

    if not pdf_paths:
        log.info("No PDFs found in knowledge base directory. Skipping auto-ingestion.")
        return

    log.info(f"Found {len(pdf_paths)} PDFs in knowledge base. Starting ingestion …")

    all_chunks = []
    knowledge_files = []

    for pdf_path in pdf_paths:
        try:
            log.info(f"  Processing: {pdf_path.name}")
            blocks = extract_text_from_pdf(str(pdf_path))
            chunks = chunk_blocks(blocks, source_name=pdf_path.stem)
            if chunks:
                all_chunks.extend(chunks)
                knowledge_files.append({
                    "filename": pdf_path.name,
                    "num_chunks": len(chunks),
                    "source": "preloaded",
                })
                log.info(f"  ✓ {pdf_path.name}: {len(blocks)} blocks → {len(chunks)} chunks")
            else:
                log.warning(f"  ✗ {pdf_path.name}: no text extracted")
        except Exception as e:
            log.error(f"  ✗ {pdf_path.name}: {e}")

    if all_chunks:
        index = build_faiss_index(all_chunks)
        sessions[KNOWLEDGE_SESSION_ID] = {
            "index": index,
            "chunks": all_chunks,
            "filename": "Knowledge Base",
        }
        save_vector_store()
        log.info(f"Knowledge base ready: {len(knowledge_files)} files, {len(all_chunks)} total chunks")
    else:
        log.warning("No chunks extracted from knowledge base PDFs.")


def add_files_to_knowledge_base(pdf_paths_and_names: list[tuple], source: str = "uploaded"):
    """Add new PDF files to the existing knowledge base session and persist."""
    global knowledge_files

    all_new_chunks = []
    processed = []
    failed = []

    for pdf_path, original_name in pdf_paths_and_names:
        try:
            blocks = extract_text_from_pdf(str(pdf_path))
            chunks = chunk_blocks(blocks, source_name=Path(original_name).stem)
            if chunks:
                all_new_chunks.extend(chunks)
                knowledge_files.append({
                    "filename": original_name,
                    "num_chunks": len(chunks),
                    "source": source,
                })
                processed.append(original_name)
                log.info(f"  ✓ {original_name}: {len(chunks)} chunks")
            else:
                failed.append(original_name)
                log.warning(f"  ✗ {original_name}: no text extracted")
        except Exception as e:
            failed.append(original_name)
            log.error(f"  ✗ {original_name}: {e}")

    if not all_new_chunks:
        return processed, failed, 0

    # Merge into existing session or create new
    if KNOWLEDGE_SESSION_ID in sessions:
        existing = sessions[KNOWLEDGE_SESSION_ID]
        merged_chunks = existing["chunks"] + all_new_chunks
        index = build_faiss_index(merged_chunks)
        sessions[KNOWLEDGE_SESSION_ID] = {
            "index": index,
            "chunks": merged_chunks,
            "filename": "Knowledge Base",
        }
    else:
        index = build_faiss_index(all_new_chunks)
        sessions[KNOWLEDGE_SESSION_ID] = {
            "index": index,
            "chunks": all_new_chunks,
            "filename": "Knowledge Base",
        }

    # Persist to disk
    save_vector_store()

    return processed, failed, len(all_new_chunks)


# ─────────────────────────── PDF Parsing ────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    clean = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(" | ".join(clean))
                table_str = "\n".join(rows)
                if table_str.strip():
                    blocks.append({"type": "table", "content": table_str, "page": page_num})

            cropped = page
            for bbox in table_bboxes:
                try:
                    cropped = cropped.outside_bbox(bbox)
                except Exception:
                    pass

            text = cropped.extract_text() or ""
            if text.strip():
                blocks.append({"type": "text", "content": text, "page": page_num})

    return blocks


# ─────────────────────────── Chunking ───────────────────────────
def chunk_blocks(blocks: list[dict], source_name: str = "") -> list[str]:
    chunks = []
    for block in blocks:
        content = block["content"].strip()
        page    = block["page"]
        src_tag = f" | Report: {source_name}" if source_name else ""
        prefix  = f"[Page {page} | {block['type'].upper()}{src_tag}]\n"

        if block["type"] == "table":
            if len(content) <= CHUNK_SIZE * 2:
                chunks.append(prefix + content)
            else:
                rows = content.split("\n")
                current = prefix
                for row in rows:
                    if len(current) + len(row) + 1 > CHUNK_SIZE * 2:
                        chunks.append(current)
                        current = prefix + row + "\n"
                    else:
                        current += row + "\n"
                if current.strip():
                    chunks.append(current)
        else:
            start = 0
            while start < len(content):
                end   = start + CHUNK_SIZE
                chunk = content[start:end]
                chunks.append(prefix + chunk)
                start += CHUNK_SIZE - CHUNK_OVERLAP
                if start >= len(content):
                    break

    return chunks


# ─────────────────────────── FAISS index ────────────────────────
def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatIP:
    log.info(f"Embedding {len(chunks)} chunks …")
    embeddings = embedder.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built.")
    return index


def retrieve(query: str, session_id: str, top_k: int = TOP_K) -> list[str]:
    if session_id not in sessions:
        raise ValueError("Session not found. Please upload a document first.")
    q_emb  = embedder.encode([query], normalize_embeddings=True)
    q_emb  = np.array(q_emb, dtype=np.float32)
    index  = sessions[session_id]["index"]
    chunks = sessions[session_id]["chunks"]
    D, I   = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]


# ─────────────────────────── LLM Prompts ────────────────────────
SYSTEM_PROMPT = """You are a financial analyst AI assistant.
You answer questions strictly based on the provided context extracted from an Annual Report.
Rules:
- Always cite the source of your information using the format: (Report: <report_name>, Page <N>)
- Every factual claim or figure must include a citation.
- If multiple pages support an answer, cite all of them.
- If the answer is not in the context, say "I could not find that information in the report."
- Do not hallucinate numbers, figures, or page references.
- Be concise and structured. Use bullet points for multi-part answers.
"""

GENERAL_SYSTEM_PROMPT = """You are an expert financial analyst AI assistant with deep knowledge of:
- Financial markets, instruments, and concepts
- Accounting principles and financial statements
- Investment strategies and portfolio management
- Banking, insurance, and fintech
- Macroeconomics and monetary policy

Rules:
- Answer general finance questions directly from your knowledge.
- If a question requires a specific company's data or report (e.g. "What was HDFC's revenue?"),
  respond: "This requires an Annual Report. Please upload the relevant PDF and I'll extract the answer for you."
- Be concise, accurate, and educational.
- Never hallucinate specific figures for companies without a report uploaded.
"""

def build_prompt(question: str, context_chunks: list[str], history: list[dict], report_name: str = "Annual Report") -> str:
    context  = "\n\n---\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": (
            f"DOCUMENT NAME: {report_name}\n\n"
            f"RELEVANT CONTEXT (each section is prefixed with its page number):\n\n{context}"
        )},
    ]
    for turn in history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": question})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = ""
    for m in messages:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
    return prompt + "Assistant:"


def build_general_prompt(question: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": question})
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = ""
    for m in messages:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
    return prompt + "Assistant:"


def generate_answer(prompt: str) -> str:
    inputs    = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][input_len:]
    answer     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    del inputs, outputs
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return answer


# ─────────────────────────── Routes ─────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Financial AI API is running 🚀"}


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "device":          DEVICE,
        "model_loaded":    model is not None,
        "active_sessions": len(sessions),
        "knowledge_base_loaded": KNOWLEDGE_SESSION_ID in sessions,
        "knowledge_files": len(knowledge_files),
    }


# ── Multi-file upload ───────────────────────────────────────────
@app.post("/upload-multiple", response_model=MultiUploadResponse)
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """Upload multiple PDF or ZIP files, save to knowledge_base/, vectorize and persist."""
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths_and_names = []
    temp_zip_paths = []
    temp_dirs = []

    for file in files:
        fname = file.filename.lower()

        if fname.endswith(".pdf"):
            dest = KNOWLEDGE_BASE_DIR / file.filename
            if dest.exists():
                stem = dest.stem
                dest = KNOWLEDGE_BASE_DIR / f"{stem}_{uuid.uuid4().hex[:6]}.pdf"
            try:
                content = await file.read()
                with open(dest, "wb") as f:
                    f.write(content)
                pdf_paths_and_names.append((dest, dest.name))
                log.info(f"Saved PDF to knowledge_base: {dest.name} ({len(content)//1024} KB)")
            except Exception as e:
                log.error(f"Failed to save {file.filename}: {e}")

        elif fname.endswith(".zip"):
            zip_save = UPLOAD_DIR / f"{uuid.uuid4()}.zip"
            try:
                content = await file.read()
                with open(zip_save, "wb") as f:
                    f.write(content)
                temp_zip_paths.append(zip_save)
                log.info(f"Received ZIP: {file.filename} ({len(content)//1024} KB)")

                extract_dir = UPLOAD_DIR / f"_zip_{uuid.uuid4()}"
                extract_dir.mkdir(exist_ok=True)
                temp_dirs.append(extract_dir)

                with zipfile.ZipFile(zip_save, "r") as zf:
                    zf.extractall(extract_dir)

                for pdf in extract_dir.rglob("*.pdf"):
                    dest = KNOWLEDGE_BASE_DIR / pdf.name
                    if dest.exists():
                        stem = dest.stem
                        dest = KNOWLEDGE_BASE_DIR / f"{stem}_{uuid.uuid4().hex[:6]}.pdf"
                    shutil.copy2(pdf, dest)
                    pdf_paths_and_names.append((dest, dest.name))
                    log.info(f"  Extracted PDF to knowledge_base: {dest.name}")
            except Exception as e:
                log.error(f"Failed to process ZIP {file.filename}: {e}")
        else:
            continue

    if not pdf_paths_and_names:
        raise HTTPException(status_code=400, detail="No valid PDF files found (upload PDFs or ZIPs containing PDFs).")

    try:
        processed, failed, new_chunks = add_files_to_knowledge_base(pdf_paths_and_names, source="uploaded")

        total_chunks = 0
        if KNOWLEDGE_SESSION_ID in sessions:
            total_chunks = len(sessions[KNOWLEDGE_SESSION_ID]["chunks"])

        return MultiUploadResponse(
            session_id=KNOWLEDGE_SESSION_ID,
            total_chunks=total_chunks,
            files_processed=processed,
            files_failed=failed,
            message=f"{len(processed)} file(s) added to knowledge base. {len(failed)} failed.",
        )
    except Exception as e:
        log.exception("Multi-upload error")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        for p in temp_zip_paths:
            if p.exists():
                p.unlink()
        for d in temp_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


# ── Query ────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        effective_session = req.session_id

        if not effective_session or effective_session == "none":
            if KNOWLEDGE_SESSION_ID in sessions:
                effective_session = KNOWLEDGE_SESSION_ID
            else:
                prompt = build_general_prompt(req.question, req.history or [])
                answer = generate_answer(prompt)
                return QueryResponse(answer=answer, sources=[], session_id=req.session_id or "none")

        if effective_session not in sessions:
            prompt = build_general_prompt(req.question, req.history or [])
            answer = generate_answer(prompt)
            return QueryResponse(answer=answer, sources=[], session_id=req.session_id or "none")

        context_chunks = retrieve(req.question, effective_session)
        prompt = build_prompt(
            req.question, context_chunks, req.history or [],
            report_name=sessions[effective_session].get("filename", "Annual Report"),
        )
        answer = generate_answer(prompt)
        return QueryResponse(
            answer=answer,
            sources=[c[:200] + "…" for c in context_chunks],
            session_id=effective_session,
        )
    except Exception as e:
        log.exception("Query error")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    gc.collect()
    return {"message": f"Session {session_id} deleted."}


# ─────────────────────────── Entry point ────────────────────────
if __name__ == "__main__":
    if NGROK_TOKEN:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(NGROK_TOKEN)
            tunnel = ngrok.connect(8000, bind_tls=True)
            print(f"\n✅ Public URL : {tunnel.public_url}")
            print(f"   API docs   : {tunnel.public_url}/docs\n")
        except ImportError:
            print("pyngrok not installed — skipping ngrok tunnel")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")