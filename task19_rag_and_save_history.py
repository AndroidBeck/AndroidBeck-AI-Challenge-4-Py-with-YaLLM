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
YAGPT_MODEL = os.getenv("YAGPT_MODEL", "yandexgpt")  # or "yandexgpt"

# RAG params
TOP_K = 3                  # how many chunks to actually use for augmentation at once
TOP_K_RETRIEVE = 9         # how many chunks to retrieve from FAISS before filtering/reranking
SIMILARITY_THRESHOLD = 0.6 # minimum cosine similarity to keep a chunk


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
# YANDEX – LLM CALL (WITH DIALOG HISTORY)
# =========================

def call_llm(messages: List[dict]) -> str:
    """
    Call YandexGPT with full dialog history.

    `messages` should be a list of dicts:
    [
      {"role": "user", "text": "..."},
      {"role": "assistant", "text": "..."},
      ...
    ]

    We prepend a single system message here.
    """
    if not YAC_API_KEY or not YAC_FOLDER:
        raise RuntimeError(
            "YandexGPT config is missing. Please set YAC_FOLDER and YAC_API_KEY env vars."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
        "x-folder-id": YAC_FOLDER,
    }

    system_message = {
        "role": "system",
        "text": (
            "You are a helpful assistant. You see the full dialog history in messages. "
            "When context from local documents is provided in the latest user message, "
            "you must use it to answer the question. If the context does not contain the "
            "answer, say that you answer based on your general knowledge. "
            "Follow any instructions in the latest user message about RAG quotations."
        ),
    }

    body = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAGPT_MODEL}",
        "completionOptions": {
            "maxTokens": "2500",
            "temperature": 0.2
        },
        "messages": [system_message] + messages,
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
    We also L2-normalize rows so we can use inner-product as cosine similarity.
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

    matrix = np.vstack(vectors)  # shape: [N, dim]

    # normalize each row to unit length for cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    matrix = matrix / norms

    return np.array(chunk_ids, dtype="int64"), matrix


def build_faiss_index(matrix: np.ndarray):
    dim = matrix.shape[1]
    # We use inner product on normalized vectors → cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index


def retrieve_candidates(
    conn: sqlite3.Connection,
    query: str,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
):
    """
    Retrieve up to `top_k_retrieve` nearest chunks, sort them by similarity,
    filter by threshold.

    Returns a tuple:
      (above_threshold, all_candidates)

    where each list contains dicts:
      {
        "chunk_id": int,
        "doc_path": str,
        "doc_name": str,
        "chunk_index": int,
        "text": str,
        "score": float
      }
    """
    if faiss is None:
        return [], []

    chunk_ids, matrix = load_embeddings(conn)
    if matrix.shape[0] == 0:
        return [], []

    # Get embedding for query
    q_emb = get_embedding(query)
    q_vec = np.array(q_emb, dtype="float32").reshape(1, -1)

    # normalize query vector for cosine similarity
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    q_norm = np.clip(q_norm, 1e-8, None)
    q_vec = q_vec / q_norm

    # Build index and search
    index = build_faiss_index(matrix)  # IndexFlatIP with cosine similarity
    k = min(top_k_retrieve, matrix.shape[0])
    scores, indices = index.search(q_vec, k)

    cur = conn.cursor()
    candidates = []

    for pos, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(scores[0][pos])
        chunk_id = int(chunk_ids[idx])

        # Get metadata for this chunk
        cur.execute(
            "SELECT doc_path, chunk_index, text FROM chunks WHERE id = ?;",
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            continue

        doc_path, chunk_index, text = row
        candidates.append(
            {
                "chunk_id": chunk_id,
                "doc_path": doc_path,
                "doc_name": os.path.basename(doc_path),
                "chunk_index": chunk_index,
                "text": text,
                "score": score,  # cosine similarity (since we normalized)
            }
        )

    if not candidates:
        return [], []

    # Rerank: sort by similarity descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Debug: print all retrieved candidates with scores
    print("\n[RAG Debug] Retrieved candidates (sorted by similarity):")
    for c in candidates:
        print(f"{c['score']:.4f} — {c['doc_name']} chunk {c['chunk_index']}")

    # Filter by similarity threshold
    filtered = [c for c in candidates if c["score"] >= similarity_threshold]

    if not filtered:
        print(f"\n[RAG Debug] No chunks above threshold {similarity_threshold}.")
    else:
        print(
            f"\n[RAG] Total candidates above threshold={similarity_threshold}: "
            f"{len(filtered)} (retrieved={len(candidates)})."
        )

    return filtered, candidates


# =========================
# BUILD AUGMENTED PROMPT — LLM САМА ДЕЛАЕТ RAG QUOTATIONS
# =========================

def build_augmented_prompt(user_question: str, chunks: List[dict]) -> str:
    """
    Build a prompt that:
    - embeds context chunks with [DOC: doc_name | CHUNK: index]
    - instructs the model to add a 'RAG quotations:' section at the END
      listing only chunks that were actually used.
    """
    if not chunks:
        return user_question

    parts = []
    for ch in chunks:
        doc_name = ch.get("doc_name") or os.path.basename(ch["doc_path"])
        idx = ch["chunk_index"]
        text = ch["text"]
        parts.append(
            f"[DOC: {doc_name} | CHUNK: {idx}]\n{text}\n"
        )

    context_block = "\n\n".join(parts)

    prompt = f"""Use the following context from local documents to answer the question.

The context is split into chunks marked like:
[DOC: doc_name | CHUNK: chunk_index]

RULES FOR SOURCES:
1. Use the context when it is helpful to answer the question.
2. When you USE information from a chunk in your answer, at the END of your answer
   add a section:

RAG quotations:
- doc_name chunk chunk_index: "short quote from that chunk"
- ...

3. Include ONLY chunks that actually influenced your answer.
4. If you did not use any context, write exactly:
RAG quotations: none

=== CONTEXT START ===
{context_block}
=== CONTEXT END ===

Question: {user_question}
"""

    return prompt


# =========================
# MAIN LOOP (NOW WITH HISTORY)
# =========================

def main():
    print("=== Day 19 – RAG with dialog history (all messages sent each request) ===")
    print("Docs folder:", DOCS_DIR)
    print("DB:", DB_PATH)
    print("\nCommands:")
    print("  chunk [size] [overlap]  - (re)chunk docs and build embeddings (defaults: 1000 200)")
    print("  setrag 0/1              - disable/enable RAG augmentation (default: 0)")
    print("  exit                    - quit")
    print("Any other input will be treated as a question.\n")

    conn = init_db()
    rag_enabled = False
    history: List[dict] = []  # dialog history without system message

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
            print("Exiting script. Goodbye!")
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

        # We'll keep candidates and offset per question to allow "lower scored" retries
        above_threshold = []
        all_candidates = []
        offset = 0

        if rag_enabled and faiss is not None:
            print("\n[RAG] Retrieving context from local documents...")
            above_threshold, all_candidates = retrieve_candidates(
                conn,
                question,
                TOP_K_RETRIEVE,
                SIMILARITY_THRESHOLD
            )

            if not all_candidates:
                # No RAG candidates at all → plain question
                print("[RAG] No chunks found. Sending plain question.")
                full_prompt = question

                user_message = {"role": "user", "text": full_prompt}
                messages_for_llm = history + [user_message]

                print("\n[LLM] Sending request to YandexGPT...")
                try:
                    answer = call_llm(messages_for_llm)
                except Exception as e:
                    print(f"[LLM ERROR] {e}")
                    continue

                history.extend([
                    user_message,
                    {"role": "assistant", "text": answer}
                ])

                print("\n=== ANSWER ===")
                print(answer)
                print("\nRAG quotations: none")
                print("==============\n")
                continue

            # There ARE candidates, but maybe all are below threshold.
            if above_threshold:
                # Normal case: use only chunks above threshold for the first answer
                candidates = above_threshold
                print(
                    f"[RAG] Will use only chunks with similarity >= {SIMILARITY_THRESHOLD} "
                    f"for the first answer."
                )

                # First group of chunks (top_k)
                group = candidates[offset: offset + TOP_K]
                offset += len(group)

                print(
                    f"[RAG] Augmenting question with {len(group)} chunk(s) "
                    f"(TOP_K={TOP_K}, threshold={SIMILARITY_THRESHOLD})."
                )
                for i, c in enumerate(group, start=1):
                    preview = textwrap.shorten(
                        c["text"].strip().replace("\n", " "),
                        width=100,
                        placeholder="..."
                    )
                    print(f"  {i}. {c['doc_path']}: {preview}")

                full_prompt = build_augmented_prompt(question, group)

                user_message = {"role": "user", "text": full_prompt}
                messages_for_llm = history + [user_message]

                print("\n[LLM] Sending request to YandexGPT...")
                try:
                    answer = call_llm(messages_for_llm)
                except Exception as e:
                    print(f"[LLM ERROR] {e}")
                    continue

                history.extend([
                    user_message,
                    {"role": "assistant", "text": answer}
                ])

                print("\n=== ANSWER ===")
                print(answer)
                print("==============\n")

                # Now, as long as there are still unused ABOVE-THRESHOLD candidates,
                # offer alternative answers with other high-sim chunks.
                while rag_enabled and offset < len(candidates):
                    try:
                        choice = input(
                            "Do you want another answer with lower scored rag chunks (still above threshold)? (y/n) "
                        ).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nBye!")
                        return

                    if choice != "y":
                        break

                    # Next group of chunks
                    group = candidates[offset: offset + TOP_K]
                    if not group:
                        print("[RAG] No more chunks to try.")
                        break

                    offset += len(group)

                    print(
                        f"\n[RAG] Using another {len(group)} chunk(s) "
                        f"from lower-scored group (still above threshold)."
                    )
                    for i, c in enumerate(group, start=1):
                        preview = textwrap.shorten(
                            c["text"].strip().replace("\n", " "),
                            width=100,
                            placeholder="..."
                        )
                        print(f"  {i}. {c['doc_path']}: {preview}")

                    full_prompt = build_augmented_prompt(question, group)

                    user_message = {"role": "user", "text": full_prompt}
                    messages_for_llm = history + [user_message]

                    print("\n[LLM] Sending request to YandexGPT with other high-similarity chunks...")
                    try:
                        alt_answer = call_llm(messages_for_llm)
                    except Exception as e:
                        print(f"[LLM ERROR] {e}")
                        break

                    history.extend([
                        user_message,
                        {"role": "assistant", "text": alt_answer}
                    ])

                    print("\n=== ALTERNATIVE ANSWER ===")
                    print(alt_answer)
                    print("================================\n")

                # После использования всех или части above-threshold чанков
                # можно опционально предложить below-threshold, если они есть.
                if len(all_candidates) > len(above_threshold):
                    try:
                        choice = input(
                            "Do you want to try additional answers using chunks below the similarity threshold? (y/n) "
                        ).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nBye!")
                        return

                    if choice == "y":
                        below = [c for c in all_candidates if c not in above_threshold]
                        offset = 0
                        candidates = below

                        while rag_enabled and offset < len(candidates):
                            group = candidates[offset: offset + TOP_K]
                            if not group:
                                print("[RAG] No more below-threshold chunks to try.")
                                break

                            offset += len(group)

                            print(
                                f"\n[RAG] Using {len(group)} chunk(s) "
                                f"from BELOW-threshold group."
                            )
                            for i, c in enumerate(group, start=1):
                                preview = textwrap.shorten(
                                    c["text"].strip().replace("\n", " "),
                                    width=100,
                                    placeholder="..."
                                )
                                print(f"  {i}. {c['doc_path']}: {preview}")

                            full_prompt = build_augmented_prompt(question, group)

                            user_message = {"role": "user", "text": full_prompt}
                            messages_for_llm = history + [user_message]

                            print("\n[LLM] Sending request to YandexGPT with below-threshold chunks...")
                            try:
                                alt_answer = call_llm(messages_for_llm)
                            except Exception as e:
                                print(f"[LLM ERROR] {e}")
                                break

                            history.extend([
                                user_message,
                                {"role": "assistant", "text": alt_answer}
                            ])

                            print("\n=== ALTERNATIVE ANSWER (below threshold) ===")
                            print(alt_answer)
                            print("============================================\n")

            else:
                # No chunks above threshold at all:
                #   → first answer is plain (no RAG),
                #   → afterwards we offer to try low-similarity chunks.
                candidates = all_candidates
                print(
                    f"[RAG] No chunks above threshold {SIMILARITY_THRESHOLD}. "
                    "First answer will NOT use RAG context."
                )

                # First, plain answer
                full_prompt = question

                user_message = {"role": "user", "text": full_prompt}
                messages_for_llm = history + [user_message]

                print("\n[LLM] Sending request to YandexGPT (plain question, no RAG)...")
                try:
                    answer = call_llm(messages_for_llm)
                except Exception as e:
                    print(f"[LLM ERROR] {e}")
                    continue

                history.extend([
                    user_message,
                    {"role": "assistant", "text": answer}
                ])

                print("\n=== ANSWER ===")
                print(answer)
                print("\nRAG quotations: none")
                print("==============\n")

                # Now offer to try low-similarity chunks for alternative answers
                offset = 0
                while rag_enabled and offset < len(candidates):
                    try:
                        choice = input(
                            "Do you want another answer using low-similarity rag chunks from local docs? (y/n) "
                        ).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nBye!")
                        return

                    if choice != "y":
                        break

                    group = candidates[offset: offset + TOP_K]
                    if not group:
                        print("[RAG] No more chunks to try.")
                        break

                    offset += len(group)

                    print(
                        f"\n[RAG] Using {len(group)} low-similarity chunk(s) "
                        f"from local documents."
                    )
                    for i, c in enumerate(group, start=1):
                        preview = textwrap.shorten(
                            c["text"].strip().replace("\n", " "),
                            width=100,
                            placeholder="..."
                        )
                        print(f"  {i}. {c['doc_path']}: {preview}")

                    full_prompt = build_augmented_prompt(question, group)

                    user_message = {"role": "user", "text": full_prompt}
                    messages_for_llm = history + [user_message]

                    print("\n[LLM] Sending request to YandexGPT with low-similarity chunks...")
                    try:
                        alt_answer = call_llm(messages_for_llm)
                    except Exception as e:
                        print(f"[LLM ERROR] {e}")
                        break

                    history.extend([
                        user_message,
                        {"role": "assistant", "text": alt_answer}
                    ])

                    print("\n=== ALTERNATIVE ANSWER (low similarity) ===")
                    print(alt_answer)
                    print("===========================================\n")

        else:
            # RAG is disabled or faiss not available
            if rag_enabled and faiss is None:
                print("[WARN] RAG requested, but faiss not installed → using plain question.")
            else:
                print("\n[RAG] RAG is OFF. Sending plain question.")

            full_prompt = question

            user_message = {"role": "user", "text": full_prompt}
            messages_for_llm = history + [user_message]

            print("\n[LLM] Sending request to YandexGPT...")
            try:
                answer = call_llm(messages_for_llm)
            except Exception as e:
                print(f"[LLM ERROR] {e}")
                continue

            history.extend([
                user_message,
                {"role": "assistant", "text": answer}
            ])

            print("\n=== ANSWER ===")
            print(answer)
            print("\nRAG quotations: none")
            print("==============\n")


if __name__ == "__main__":
    main()
