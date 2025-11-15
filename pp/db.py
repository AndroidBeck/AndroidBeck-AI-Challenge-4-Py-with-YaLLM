# db.py
import sqlite3
import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = "pp_memory.db"


# =========================
# DB CONNECTION & INIT
# =========================

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,              -- 'system' | 'user' | 'assistant' | 'summary'
        content TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        is_summary INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        request_id TEXT,
        input_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    )
    """)

    conn.commit()
    conn.close()


# =========================
# CONVERSATIONS
# =========================

def create_conversation(title: Optional[str] = None) -> int:
    if title is None:
        title = (
            "Conversation started at "
            + datetime.datetime.now().isoformat(timespec="seconds")
        )

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO conversations (title) VALUES (?)
    """, (title,))

    conversation_id = cur.lastrowid

    conn.commit()
    conn.close()

    return conversation_id


def get_last_conversation_with_active_messages() -> Optional[Tuple[int, str, str, str]]:
    """
    Returns the last conversation that has at least one active message.
    If none, returns None.

    Result: (id, title, created_at, updated_at) or None
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT c.id, c.title, c.created_at, c.updated_at
        FROM conversations c
        JOIN messages m ON m.conversation_id = c.id AND m.is_active = 1
        GROUP BY c.id
        ORDER BY c.updated_at DESC, c.id DESC
        LIMIT 1
    """)

    row = cur.fetchone()

    conn.close()
    return row


def touch_conversation(conversation_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (conversation_id,))

    conn.commit()
    conn.close()


# =========================
# MESSAGES
# =========================

def save_message(
    conversation_id: int,
    role: str,
    content: str,
    is_summary: bool = False,
    is_active: bool = True,
) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO messages (conversation_id, role, content, is_summary, is_active)
        VALUES (?, ?, ?, ?, ?)
    """, (
        conversation_id,
        role,
        content,
        int(is_summary),
        int(is_active),
    ))

    conn.commit()
    conn.close()

    touch_conversation(conversation_id)


def load_active_messages(conversation_id: int) -> List[Dict[str, str]]:
    """
    Returns messages as a list of dicts:
    [{"role": ..., "text": ...}, ...]
    suitable for Yandex's 'messages' field (after adding system prompt).
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT role, content
        FROM messages
        WHERE conversation_id = ? AND is_active = 1
        ORDER BY id ASC
    """, (conversation_id,))

    rows = cur.fetchall()
    conn.close()

    return [{"role": role, "text": content} for (role, content) in rows]


def archive_non_summary_messages(conversation_id: int) -> None:
    """
    Marks all non-summary messages of this conversation as inactive.
    We keep them in DB for history but they won't go into context.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE messages
        SET is_active = 0
        WHERE conversation_id = ? AND is_summary = 0
    """, (conversation_id,))

    conn.commit()
    conn.close()


def deactivate_all_messages_for_conversation(conversation_id: int) -> None:
    """
    Marks ALL messages of this conversation as inactive (including summaries).
    Used when starting a completely new topic with /new if you want that behavior.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE messages
        SET is_active = 0
        WHERE conversation_id = ?
    """, (conversation_id,))

    conn.commit()
    conn.close()


# =========================
# STATS
# =========================

def save_stats(
    conversation_id: int,
    usage: Dict[str, Any],
    request_id: Optional[str] = None,
) -> None:
    if not usage:
        return

    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")

    # Yandex sometimes returns strings, so convert safely
    def to_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO stats (conversation_id, request_id, input_tokens, completion_tokens, total_tokens)
        VALUES (?, ?, ?, ?, ?)
    """, (
        conversation_id,
        request_id,
        to_int(input_tokens),
        to_int(completion_tokens),
        to_int(total_tokens),
    ))

    conn.commit()
    conn.close()
