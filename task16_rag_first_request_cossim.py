import os
import sqlite3
import time
import textwrap
from typing import List, Tuple

import requests
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except ImportError:
    faiss = None
    print("WARNING: faiss not installed. Install with `pip install faiss-cpu` "
          "for fast similarity search. For now, RAG will be disabled.")


# =========================
# CONFIG
# =========================

DB_PATH = "day16_rag.sqlite"
DOCS_DIR = "docs"

# Ollama embeddings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

# YandexGPT config
YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")
YAGPT_MODEL = os.getenv("YAGPT_MODEL", "yandexgpt-lite")  # or "yandexgpt"

TOP_K = 5  # how many chunks to retrieve for RAG


# =========================
# DB SETUP
# =========================

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    return conn


# =========================
# OLLAMA – EMBEDDINGS
# =========================

def get_embedding(text: str) -> List[float]:
    """
    Call Ollama embeddings endpoint to get a vector for given text.
    Requires `ollama serve` + `ollama pull nomic-embed-text`.
    """
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embedding": [...]}
    return data["embedding"]


# =========================
# YANDEX – LLM CALL
# =========================

def call_llm(prompt: str) -> str:
    if not YAC_API_KEY or not YAC_FOLDER:
        raise RuntimeError(
            "YandexGPT config is missing. Please set YAC_FOLDER and YAC_API_KEY env vars."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
        "x-folder-id": YAC_FOLDER,
    }

    body = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAGPT_MODEL}",
        "completionOptions": {
            "maxTokens": "800",
            "temperature": 0.2
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "You are a helpful assistant. When context from local documents is "
                    "provided, you must use it to answer the question. If context does not "
                    "contain the answer, say that you answer based on your general knowledge."
                ),
            },
            {
                "role": "user",
                "text": prompt,
            },
        ],
    }

    resp = requests.post(YAGPT_URL, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # YandexGPT usually returns: result -> alternatives -> message -> text
    try:
        alts = data["result"]["alternatives"]
        texts = [alt["message"]["text"] for alt in alts if "message" in alt]
        return "\n".join(texts).strip()
    except Exception:
        # Fallback: return whole json as string if format unexpected
        return f"Unexpected LLM response format:\n{data}"


# =========================
# CHUNKING UTILS
# =========================

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple character-based sliding-window chunking.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    stride = max(1, size - overlap)

    while start < n:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start += stride

    return chunks


def iter_docs(doc_dir: str = DOCS_DIR) -> List[str]:
    paths = []
    if not os.path.isdir(doc_dir):
        return paths

    for root, _, files in os.walk(doc_dir):
        for fn in files:
            if fn.lower().endswith((".txt", ".md")):
                paths.append(os.path.join(root, fn))
    return sorted(paths)


# =========================
# CHUNK COMMAND
# =========================

def command_chunk(conn: sqlite3.Connection, chunk_size: int = 1000, overlap: int = 200) -> None:
    if faiss is None:
        print("RAG disabled: faiss is not installed. Install `faiss-cpu` first.")
        return

    docs = iter_docs()
    if not docs:
        print(f"No .txt/.md docs found in '{DOCS_DIR}/'.")
        return

    print(f"Rebuilding chunks & embeddings from {len(docs)} docs...")
    print("Clearing old data from DB...")
    conn.execute("DELETE FROM embeddings;")
    conn.execute("DELETE FROM chunks;")
    conn.commit()

    cur = conn.cursor()
    total_chunks = 0

    for doc_path in docs:
        print(f"\n[DOC] {doc_path}")
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text, size=chunk_size, overlap=overlap)
        print(f"  -> {len(chunks)} chunks")

        for idx, ch in enumerate(chunks):
            cur.execute(
                "INSERT INTO chunks (doc_path, chunk_index, text) VALUES (?, ?, ?)",
                (doc_path, idx, ch),
            )
            chunk_id = cur.lastrowid

            # embed and store as BLOB(float32)
            emb = get_embedding(ch)
            vec = np.array(emb, dtype="float32")
            blob = vec.tobytes()
            conn.execute(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )

            total_chunks += 1

        conn.commit()

    print(f"\nDone. Total chunks stored: {total_chunks}")


# =========================
# RAG RETRIEVAL
# =========================

def load_embeddings(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (chunk_ids, matrix) where matrix is shape [N, dim] float32.
    """
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, embedding FROM embeddings ORDER BY chunk_id;")
    rows = cur.fetchall()
    if not rows:
        return np.array([]), np.zeros((0, 0), dtype="float32")

    chunk_ids = []
    vectors = []
    for chunk_id, blob in rows:
        vec = np.frombuffer(blob, dtype="float32")
        chunk_ids.append(chunk_id)
        vectors.append(vec)

    matrix = np.vstack(vectors)
    return np.array(chunk_ids, dtype="int64"), matrix


def build_faiss_index(matrix: np.ndarray):
    dim = matrix.shape[1]
    # inner product works fine for normalized embeddings; here we just use L2
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    return index


def retrieve_context(conn: sqlite3.Connection, query: str, top_k: int = TOP_K) -> List[Tuple[str, str]]:
    """
    Returns a list of (doc_path, text) for top_k chunks.
    """
    if faiss is None:
        return []

    chunk_ids, matrix = load_embeddings(conn)
    if matrix.shape[0] == 0:
        return []

    # Get embedding for query
    q_emb = get_embedding(query)
    q_vec = np.array(q_emb, dtype="float32").reshape(1, -1)

    index = build_faiss_index(matrix)
    distances, indices = index.search(q_vec, min(top_k, matrix.shape[0]))

    retrieved = []
    cur = conn.cursor()
    for idx in indices[0]:
        if idx < 0:
            continue
        chunk_id = int(chunk_ids[idx])
        cur.execute(
            "SELECT doc_path, text FROM chunks WHERE id = ?;",
            (chunk_id,),
        )
        row = cur.fetchone()
        if row:
            retrieved.append((row[0], row[1]))

    return retrieved


def build_augmented_prompt(user_question: str, contexts: List[Tuple[str, str]]) -> str:
    if not contexts:
        return user_question

    parts = []
    for i, (path, text) in enumerate(contexts, start=1):
        snippet = textwrap.shorten(text.replace("\n", " "), width=400, placeholder="...")
        parts.append(f"Source {i}: {path}\n{text}\n")

    context_block = "\n\n".join(parts)

    prompt = f"""Use the following context from local documents to answer the question.

=== CONTEXT START ===
{context_block}
=== CONTEXT END ===

Question: {user_question}
"""

    return prompt


# =========================
# MAIN LOOP
# =========================

def main():
    print("=== Day 16 – The first RAG-request ===")
    print("Docs folder:", DOCS_DIR)
    print("DB:", DB_PATH)
    print("\nCommands:")
    print("  chunk [size] [overlap]  - (re)chunk docs and build embeddings (defaults: 1000 200)")
    print("  setrag 0/1              - disable/enable RAG augmentation (default: 0)")
    print("  exit                    - quit")
    print("Any other input will be treated as a question.\n")

    conn = init_db()
    rag_enabled = False

    if faiss is None:
        print("NOTE: faiss not available → RAG will be effectively OFF even if you setrag 1.\n")

    while True:
        try:
            user_input = input("Your input> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # exit
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting Day 16 script. Goodbye!")
            break

        # chunk command: "chunk", "chunk 1000", "chunk 1200 200"
        if user_input.lower().startswith("chunk"):
            parts = user_input.split()
            size = 1000
            overlap = 200
            if len(parts) >= 2:
                try:
                    size = int(parts[1])
                except ValueError:
                    print("Invalid chunk size, using default 1000.")
            if len(parts) >= 3:
                try:
                    overlap = int(parts[2])
                except ValueError:
                    print("Invalid overlap, using default 200.")

            print(f"\n[CHUNK] size={size}, overlap={overlap}")
            started = time.time()
            command_chunk(conn, size, overlap)
            print(f"[CHUNK] done in {time.time() - started:.1f} s.\n")
            continue

        # setrag command
        if user_input.lower().startswith("setrag"):
            parts = user_input.split()
            if len(parts) < 2 or parts[1] not in {"0", "1"}:
                print("Usage: setrag 0 or setrag 1")
            else:
                rag_enabled = parts[1] == "1"
                state = "ON" if rag_enabled else "OFF"
                print(f"RAG is now {state}.")
            continue

        # Otherwise – treat as a question
        question = user_input

        if rag_enabled and faiss is not None:
            print("\n[RAG] Retrieving context from local documents...")
            ctx = retrieve_context(conn, question, TOP_K)
            if not ctx:
                print("[RAG] No chunks found. Sending plain question.")
                full_prompt = question
            else:
                print(f"[RAG] Found {len(ctx)} relevant chunk(s). Using them to augment the question.")
                for i, (path, text) in enumerate(ctx, start=1):
                    preview = textwrap.shorten(text.strip().replace("\n", " "), width=100, placeholder="...")
                    print(f"  {i}. {path}: {preview}")
                full_prompt = build_augmented_prompt(question, ctx)
        else:
            if rag_enabled and faiss is None:
                print("[WARN] RAG requested, but faiss not installed → using plain question.")
            else:
                print("\n[RAG] RAG is OFF. Sending plain question.")
            full_prompt = question

        print("\n[LLM] Sending request to YandexGPT...")
        try:
            answer = call_llm(full_prompt)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            continue

        print("\n=== ANSWER ===")
        print(answer)
        print("==============\n")


if __name__ == "__main__":
    main()
