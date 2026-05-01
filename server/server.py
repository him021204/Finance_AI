"""
Financial AI Backend — FastAPI + RAG Pipeline
Model: Himanshu2124/qwen-finance-7b (HuggingFace)
Designed to run on Kaggle and expose via ngrok / cloudflared
"""

import os
import gc
import re
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Optional

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
MODEL_ID        = "Himanshu2124/qwen-finance-7b"
EMBED_MODEL_ID  = "BAAI/bge-small-en-v1.5"   # lightweight, finance-friendly embedder
CHUNK_SIZE      = 512     # characters per text chunk
CHUNK_OVERLAP   = 64      # character overlap between chunks
TOP_K           = 5       # number of chunks to retrieve per query
MAX_NEW_TOKENS  = 512
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR      = Path(tempfile.mkdtemp())

log.info(f"Using device: {DEVICE}")

# ─────────────────────────── App ────────────────────────────────
app = FastAPI(
    title="Financial AI API",
    description="RAG-powered Q&A over Annual Reports using Qwen-Finance-7B",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Global state ───────────────────────
# Holds per-session RAG state: { session_id: { "index": faiss_index, "chunks": [...] } }
sessions: dict[str, dict] = {}

# Models loaded once at startup
embedder:  SentenceTransformer = None
tokenizer: AutoTokenizer        = None
model:     AutoModelForCausalLM = None


# ─────────────────────────── Pydantic schemas ───────────────────
class QueryRequest(BaseModel):
    session_id: str
    question:   str
    history:    Optional[list[dict]] = []   # [{"role": "user"|"assistant", "content": "..."}]


class QueryResponse(BaseModel):
    answer:    str
    sources:   list[str]       # the retrieved chunk snippets used
    session_id: str


class UploadResponse(BaseModel):
    session_id:   str
    num_chunks:   int
    message:      str


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


# ─────────────────────────── PDF Parsing ────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts both plain text paragraphs and tables from a PDF.
    Returns a list of {"type": "text"|"table", "content": str, "page": int}
    """
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # ── Tables ──────────────────────────────────────────
            tables = page.extract_tables()
            table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

            for table in tables:
                if not table:
                    continue
                # Render table as a readable string  (header | row | row …)
                rows = []
                for row in table:
                    clean = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(" | ".join(clean))
                table_str = "\n".join(rows)
                if table_str.strip():
                    blocks.append({"type": "table", "content": table_str, "page": page_num})

            # ── Plain text (excluding table regions) ────────────
            # crop away table bboxes so we don't double-index
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
def chunk_blocks(blocks: list[dict]) -> list[str]:
    """
    Splits text blocks into overlapping character-level chunks.
    Tables are kept whole (or split only if very large).
    Returns list of chunk strings.
    """
    chunks = []

    for block in blocks:
        content = block["content"].strip()
        page    = block["page"]
        prefix  = f"[Page {page} | {block['type'].upper()}]\n"

        if block["type"] == "table":
            # Keep tables together; split only if > 2×CHUNK_SIZE
            if len(content) <= CHUNK_SIZE * 2:
                chunks.append(prefix + content)
            else:
                # Split table row-by-row into manageable pieces
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

        else:  # text — sliding window
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
    """Embed chunks and build an inner-product (cosine) FAISS index."""
    log.info(f"Embedding {len(chunks)} chunks …")
    embeddings = embedder.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built.")
    return index


def retrieve(query: str, session_id: str, top_k: int = TOP_K) -> list[str]:
    """Retrieve top_k most relevant chunks for a query."""
    if session_id not in sessions:
        raise ValueError("Session not found. Please upload a document first.")

    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    index  = sessions[session_id]["index"]
    chunks = sessions[session_id]["chunks"]

    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]


# ─────────────────────────── LLM Generation ─────────────────────
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

def build_prompt(question: str, context_chunks: list[str], history: list[dict], report_name: str = "Annual Report") -> str:

    """Build a chat-style prompt with retrieved context and conversation history."""
    context = "\n\n---\n\n".join(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject retrieved context as a system-level message
    messages.append({
        "role": "system",
        "content": (
            f"DOCUMENT NAME: {report_name}\n\n"
            f"RELEVANT CONTEXT (each section is prefixed with its page number):\n\n{context}"
        )
    })

    # Prior conversation turns (last 6 to keep context window manageable)
    for turn in history[-6:]:
        messages.append(turn)

    # Current question
    messages.append({"role": "user", "content": question})

    # Use tokenizer's chat template if available (Qwen supports it)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback manual formatting
        prompt = ""
        for m in messages:
            role = m["role"].capitalize()
            prompt += f"{role}: {m['content']}\n"
        prompt += "Assistant:"

    return prompt


def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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

    # Decode only the newly generated tokens
    new_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
        "status":        "ok",
        "device":        DEVICE,
        "model_loaded":  model is not None,
        "active_sessions": len(sessions),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload an Annual Report PDF.
    Parses it, chunks text + tables, builds a FAISS index, returns a session_id.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    save_path  = UPLOAD_DIR / f"{session_id}.pdf"

    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        log.info(f"[{session_id}] Saved PDF ({len(content)//1024} KB)")

        log.info(f"[{session_id}] Extracting text & tables …")
        blocks = extract_text_from_pdf(str(save_path))
        log.info(f"[{session_id}] Extracted {len(blocks)} blocks")

        chunks = chunk_blocks(blocks)
        log.info(f"[{session_id}] Created {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract any text from the PDF.")

        index = build_faiss_index(chunks)

        sessions[session_id] = {
            "index":    index,
            "chunks":   chunks,
            "filename": file.filename,
        }

        return UploadResponse(
            session_id=session_id,
            num_chunks=len(chunks),
            message=f"'{file.filename}' processed successfully. Ready for queries.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[{session_id}] Upload error")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if save_path.exists():
            save_path.unlink()  # clean up temp file


@app.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    """
    Ask a question about the uploaded Annual Report.
    Retrieves relevant chunks via FAISS, then generates an answer with the LLM.
    """
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a document first.",
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        log.info(f"[{req.session_id}] Query: {req.question[:80]}")

        # Retrieve
        context_chunks = retrieve(req.question, req.session_id)
        log.info(f"[{req.session_id}] Retrieved {len(context_chunks)} chunks")

        # Generate
        # prompt = build_prompt(req.question, context_chunks, req.history or [])
        report_name = sessions[req.session_id].get("filename", "Annual Report")
        prompt = build_prompt(req.question, context_chunks, req.history or [], report_name=report_name)
        answer = generate_answer(prompt)

        # Return short source snippets (first 200 chars each) for frontend attribution
        sources = [c[:200] + "…" for c in context_chunks]

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=req.session_id,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception(f"[{req.session_id}] Query error")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Free memory for a session when the user is done."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    gc.collect()
    return {"message": f"Session {session_id} deleted."}


# ─────────────────────────── Entry point ────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)