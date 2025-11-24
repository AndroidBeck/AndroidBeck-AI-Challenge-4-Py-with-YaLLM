import os
import json
import argparse
import pathlib
import sqlite3
import textwrap
from typing import List, Dict, Any, Optional, Iterable, Tuple

import requests

# Optional FAISS support
try:
    import faiss  # pip install faiss-cpu
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# =========================
# CONFIG
# =========================

# DOCS directory (you can override via CLI)
DEFAULT_DOCS_DIR = "docs"
DEFAULT_INDEX_DIR = "indexes"

# OpenAI embeddings config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

# Chunking config (chars, not tokens – simple but works fine for now)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File extensions we try to read as text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst",
    ".py", ".kt", ".java", ".js", ".ts",
    ".json", ".html", ".css",
    ".csv", ".tsv", ".sql",
    ".xml", ".yml", ".yaml",
}

PDF_EXTENSIONS = {".pdf"}


# =========================
# UTIL: LOADING FILE CONTENT
# =========================

def read_text_file(path: pathlib.Path) -> str:
    """Read a text file with fallback encodings."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def read_pdf_file(path: pathlib.Path) -> str:
    """Read a PDF file using PyPDF2 (if installed)."""
    try:
        import PyPDF2  # pip install PyPDF2
    except ImportError:
        print(f"[WARN] PyPDF2 not installed, skipping PDF: {path}")
        return ""

    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)


def load_document(path: pathlib.Path) -> str:
    """Load text from a file based on extension."""
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return read_text_file(path)
    elif ext in PDF_EXTENSIONS:
        return read_pdf_file(path)
    else:
        # Unknown extension – skip or treat as text
        print(f"[WARN] Skipping unsupported file type: {path}")
        return ""


# =========================
# UTIL: CHUNKING
# =========================

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    """
    Simple character-based chunking with overlap.

    Returns list of (chunk_text, start_index, end_index).
    """
    chunks = []
    text = text.strip()
    if not text:
        return chunks

    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        # Move start with overlap
        if end == n:
            break
        start = end - overlap

    return chunks


# =========================
# UTIL: EMBEDDINGS (OpenAI)
# =========================

def get_embedding_openai(text: str) -> List[float]:
    """
    Fetch embedding from OpenAI Embeddings API.
    You must have OPENAI_API_KEY set.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables")

    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": text,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["data"][0]["embedding"]


# Wrapper for possible future providers
def get_embedding(text: str, provider: str = "openai") -> List[float]:
    provider = provider.lower()
    if provider == "openai":
        return get_embedding_openai(text)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


# =========================
# INDEX WRITERS
# =========================

class BaseIndexWriter:
    def add(self, item: Dict[str, Any]):
        raise NotImplementedError

    def close(self):
        pass


class JsonIndexWriter(BaseIndexWriter):
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]):
        self.items.append(item)

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON index saved to: {self.path}")


class SqliteIndexWriter(BaseIndexWriter):
    """
    Simple SQLite index storing:
      - doc_path
      - chunk_index
      - content
      - embedding_json (TEXT)
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_path ON chunks (doc_path)"
        )
        self.conn.commit()

    def add(self, item: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks (doc_path, chunk_index, start_char, end_char, content, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item["doc_path"],
                item["chunk_index"],
                item.get("start_char"),
                item.get("end_char"),
                item["content"],
                json.dumps(item["embedding"]),
            ),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
        print(f"[OK] SQLite index saved to: {self.path}")


class FaissIndexWriter(BaseIndexWriter):
    """
    FAISS index + separate metadata JSON.
    We keep:
      - FAISS index (vectors)
      - metadata list with doc_path, chunk_index, content ranges
    """

    def __init__(self, index_path: pathlib.Path, meta_path: pathlib.Path):
        if not HAS_FAISS:
            raise RuntimeError("faiss is not installed. Install with `pip install faiss-cpu`.")

        self.index_path = index_path
        self.meta_path = meta_path
        self.vectors: List[List[float]] = []
        self.metas: List[Dict[str, Any]] = []
        self.dim: Optional[int] = None

    def add(self, item: Dict[str, Any]):
        emb = item["embedding"]
        if self.dim is None:
            self.dim = len(emb)
        elif len(emb) != self.dim:
            raise ValueError("Inconsistent embedding dimension in FAISS index")

        self.vectors.append(emb)
        meta = {
            "doc_path": item["doc_path"],
            "chunk_index": item["chunk_index"],
            "start_char": item.get("start_char"),
            "end_char": item.get("end_char"),
        }
        self.metas.append(meta)

    def close(self):
        if not self.vectors:
            print("[WARN] No vectors added, skipping FAISS index save.")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy
        vecs = np.array(self.vectors, dtype="float32")

        # Use inner-product index (you can normalize vectors for cosine)
        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs)
        faiss.write_index(index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metas, f, ensure_ascii=False, indent=2)

        print(f"[OK] FAISS index saved to: {self.index_path}")
        print(f"[OK] FAISS metadata saved to: {self.meta_path}")


# =========================
# MAIN PIPELINE
# =========================

def iter_documents(root_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield all candidate document files under root_dir."""
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in TEXT_EXTENSIONS or ext in PDF_EXTENSIONS:
            yield path


def build_index(
    docs_dir: pathlib.Path,
    index_type: str,
    embeddings_provider: str,
    index_dir: pathlib.Path,
):
    index_type = index_type.lower()
    index_dir.mkdir(parents=True, exist_ok=True)

    # Prepare writers
    writers: List[BaseIndexWriter] = []

    if index_type == "json":
        json_path = index_dir / "day15_index.json"
        writers.append(JsonIndexWriter(json_path))
    elif index_type == "sqlite":
        sqlite_path = index_dir / "day15_index.sqlite3"
        writers.append(SqliteIndexWriter(sqlite_path))
    elif index_type == "faiss":
        faiss_path = index_dir / "day15_index.faiss"
        meta_path = index_dir / "day15_index_meta.json"
        writers.append(FaissIndexWriter(faiss_path, meta_path))
    elif index_type == "all":
        json_path = index_dir / "day15_index.json"
        sqlite_path = index_dir / "day15_index.sqlite3"
        faiss_path = index_dir / "day15_index.faiss"
        meta_path = index_dir / "day15_index_meta.json"

        writers.append(JsonIndexWriter(json_path))
        writers.append(SqliteIndexWriter(sqlite_path))
        if HAS_FAISS:
            writers.append(FaissIndexWriter(faiss_path, meta_path))
        else:
            print("[WARN] faiss not installed – FAISS index will NOT be created.")
    else:
        raise ValueError("index_type must be one of: json, sqlite, faiss, all")

    try:
        doc_paths = list(iter_documents(docs_dir))
        print(f"[INFO] Found {len(doc_paths)} document(s) in {docs_dir}")

        global_chunk_id = 0

        for doc_idx, doc_path in enumerate(doc_paths, start=1):
            print(f"\n[DOC {doc_idx}/{len(doc_paths)}] {doc_path}")
            text = load_document(doc_path)
            if not text.strip():
                print("  -> Empty/unsupported document, skipping.")
                continue

            chunks = chunk_text(text)
            print(f"  -> {len(chunks)} chunk(s)")

            for chunk_index, (chunk_text_val, start_char, end_char) in enumerate(chunks):
                global_chunk_id += 1
                # Maybe small preview for logs
                preview = textwrap.shorten(chunk_text_val.replace("\n", " "), width=80)

                print(f"    [Chunk #{global_chunk_id}] doc_chunk_index={chunk_index}, "
                      f"chars={start_char}-{end_char}, preview='{preview}'")

                embedding = get_embedding(chunk_text_val, provider=embeddings_provider)

                item = {
                    "id": global_chunk_id,
                    "doc_path": str(doc_path),
                    "chunk_index": chunk_index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "content": chunk_text_val,
                    "embedding": embedding,
                }

                for w in writers:
                    w.add(item)

        print("\n[INFO] Finished processing all documents.")
    finally:
        for w in writers:
            w.close()


def main():
    parser = argparse.ArgumentParser(
        description="Day 15 – Documents indexing: chunking + embeddings + local index."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=DEFAULT_DOCS_DIR,
        help="Directory with documents to index (default: ./docs)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="Directory where the index files will be stored (default: ./indexes)",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="sqlite",
        choices=["json", "sqlite", "faiss", "all"],
        help="Which index type to build (default: all)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Embeddings provider (currently only 'openai' implemented)",
    )

    args = parser.parse_args()

    docs_dir = pathlib.Path(args.docs_dir)
    index_dir = pathlib.Path(args.index_dir)

    if not docs_dir.exists():
        print(f"[ERROR] Docs dir does not exist: {docs_dir}")
        return

    print("=== Day 15 – Documents Indexing ===")
    print(f"Docs directory  : {docs_dir}")
    print(f"Index directory : {index_dir}")
    print(f"Index type      : {args.index_type}")
    print(f"Embeddings prov.: {args.provider}")
    print(f"Embedding model : {OPENAI_EMBEDDING_MODEL}")
    print("===============================")

    build_index(
        docs_dir=docs_dir,
        index_type=args.index_type,
        embeddings_provider=args.provider,
        index_dir=index_dir,
    )


if __name__ == "__main__":
    main()
