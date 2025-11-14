import os
import json
import sqlite3
import datetime
import requests

# =========================
# YandexGPT CONFIG
# =========================

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")
# Prefer full yandexgpt over lite in this AI challenge
YAC_MODEL = os.getenv("YAC_MODEL", "yandexgpt")

BASE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer clearly and concisely. "
    "The user and assistant messages below represent an ongoing conversation."
)

SUMMARY_SYSTEM_PROMPT_TEMPLATE = (
    "You are a summarization assistant. Your task is to compress the following "
    "conversation into a single message no longer than approximately {max_tokens} tokens. "
    "Preserve all important facts, decisions, TODOs and the overall context. "
    "The summary should be self-contained and written as if it's the conversation history "
    "so far, not as bullet points about it."
)

# =========================
# SQLite CONFIG
# =========================

DB_PATH = "ai_memory_day9.db"


# =========================
# DB HELPERS
# =========================

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
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


def create_conversation(title=None):
    if title is None:
        title = "Conversation started at " + datetime.datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (title) VALUES (?)
    """, (title,))
    conversation_id = cur.lastrowid
    conn.commit()
    conn.close()
    return conversation_id


def get_last_conversation_with_active_messages():
    """
    Returns the last conversation that has at least one active message.
    If none, returns None.
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
    return row  # (id, title, created_at, updated_at) or None


def touch_conversation(conversation_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (conversation_id,))
    conn.commit()
    conn.close()


def save_message(conversation_id, role, content, is_summary=False, is_active=True):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO messages (conversation_id, role, content, is_summary, is_active)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, role, content, int(is_summary), int(is_active)))
    conn.commit()
    conn.close()
    touch_conversation(conversation_id)


def load_active_messages(conversation_id):
    """
    Returns messages as a list of dicts: [{"role": ..., "text": ...}, ...]
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


def archive_non_summary_messages(conversation_id):
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


def deactivate_all_messages_for_conversation(conversation_id):
    """
    Marks ALL messages of this conversation as inactive (including summaries).
    Used when starting a completely new topic with /new.
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


def save_stats(conversation_id, usage, request_id=None):
    if not usage:
        return
    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")

    # Yandex sometimes returns strings, so convert safely
    def to_int(x):
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


# =========================
# LLM CALL
# =========================

def call_yandex_llm(messages, temperature=0.6, max_tokens=1500):
    """
    messages: list of {"role": "system"|"user"|"assistant", "text": "..."}
    Returns (reply_text, usage_dict)
    """
    if not YAC_FOLDER or not YAC_API_KEY:
        raise RuntimeError("Missing YAC_FOLDER or YAC_API_KEY environment variables")

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "stream": False
        },
        "messages": messages
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}"
    }

    resp = requests.post(YAGPT_URL, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()

    # Extract text
    try:
        reply_text = data["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Yandex response format: {data}") from e

    # Get usage from either top-level or result.usage
    usage = data.get("usage") or data.get("result", {}).get("usage", {}) or {}

    return reply_text, usage


# =========================
# HIGH-LEVEL OPERATIONS
# =========================

def ask_llm_for_chat_answer(conversation_id, user_text):
    # Save user message
    save_message(conversation_id, "user", user_text)

    # Load active messages from DB and build full context
    history = load_active_messages(conversation_id)

    messages_for_api = [{"role": "system", "text": BASE_SYSTEM_PROMPT}]
    messages_for_api.extend(history)

    reply_text, usage = call_yandex_llm(messages_for_api)

    # Save assistant reply
    save_message(conversation_id, "assistant", reply_text)

    # Stats and logging
    save_stats(conversation_id, usage)
    print("\nASSISTANT:\n" + reply_text)

    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")
    print(
        f"\n[inputTextTokens = {input_tokens}, "
        f"completionTokens = {completion_tokens}, "
        f"totalTokens = {total_tokens}; "
        f"messages_sent = {len(messages_for_api)}]"
    )


def summarize_conversation(conversation_id, max_tokens_for_summary):
    # Get active messages
    history = load_active_messages(conversation_id)
    if not history:
        print("Nothing to summarize: no active messages.")
        return

    system_text = SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(
        max_tokens=max_tokens_for_summary
    )
    messages_for_api = [{"role": "system", "text": system_text}]
    messages_for_api.extend(history)

    # Use low temperature + max_tokens_for_summary for compression
    reply_text, usage = call_yandex_llm(
        messages_for_api,
        temperature=0.3,
        max_tokens=max_tokens_for_summary
    )

    # Archive all previous (non-summary) messages
    archive_non_summary_messages(conversation_id)

    # Save new summary message as active
    save_message(conversation_id, "system", reply_text, is_summary=True, is_active=True)

    save_stats(conversation_id, usage)
    print("\nSUMMARY:\n" + reply_text)
    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")
    print(
        f"\n[inputTextTokens = {input_tokens}, "
        f"completionTokens = {completion_tokens}, "
        f"totalTokens = {total_tokens}; "
        f"messages_sent = {len(messages_for_api)}]"
    )


# =========================
# MAIN LOOP
# =========================

def main():
    init_db()

    # Auto-select conversation:
    # If there is a conversation with active messages, continue it.
    # Otherwise, create a new one.
    last_active = get_last_conversation_with_active_messages()
    if last_active is not None:
        conversation_id, title, created_at, updated_at = last_active
        print(f"Continuing conversation #{conversation_id}: '{title}' "
              f"(created_at={created_at}, updated_at={updated_at})")
    else:
        conversation_id = create_conversation()
        print(f"Created new conversation #{conversation_id}")

    print("\nDay 9 – External Memory Chat")
    print("Type your question, or commands:")
    print("  /summarize X, /sum X, /compress X  – summarize active messages into ~X tokens (default 400)")
    print("  /new                               – start a NEW conversation (new topic)")
    print("  /exit                              – quit")

    while True:
        user_input = input("\nYOU > ").strip()
        if not user_input:
            continue

        lower = user_input.lower()

        # Exit command
        if lower == "/exit":
            print("Goodbye!")
            break

        # Start a new conversation: deactivate ALL messages of current conv, then create new
        if lower == "/new":
            deactivate_all_messages_for_conversation(conversation_id)
            conversation_id = create_conversation()
            print(f"Started NEW conversation #{conversation_id}")
            continue

        # Summarization commands: /summarize, /compress, /sum
        if lower.startswith("/summarize") or lower.startswith("/compress") or lower.startswith("/sum"):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    max_tokens_for_summary = int(parts[1])
                except ValueError:
                    print("Invalid token count. Usage: /summarize 400")
                    continue
            else:
                # Default summary length
                max_tokens_for_summary = 400

            summarize_conversation(conversation_id, max_tokens_for_summary)
        else:
            # Normal chat turn
            ask_llm_for_chat_answer(conversation_id, user_input)


if __name__ == "__main__":
    main()
