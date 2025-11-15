# chat_logic.py
from typing import Any, Dict

from db import (
    archive_non_summary_messages,
    load_active_messages,
    save_message,
    save_stats,
)
from llm_client import call_yandex_llm, LlmError


# =========================
# SYSTEM PROMPTS
# =========================

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
# UTILS
# =========================

def _print_usage(usage: Dict[str, Any], messages_sent: int, label: str) -> None:
    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")

    print(
        f"\n[{label}] "
        f"inputTextTokens = {input_tokens}, "
        f"completionTokens = {completion_tokens}, "
        f"totalTokens = {total_tokens}; "
        f"messages_sent = {messages_sent}"
    )


# =========================
# HIGH-LEVEL OPERATIONS
# =========================

def ask_llm_for_chat_answer(conversation_id: int, user_text: str) -> None:
    """
    Normal chat turn:
    - save user message
    - load active history
    - call LLM with system + history
    - save assistant reply
    - log token usage
    """
    # Save user message
    save_message(conversation_id, "user", user_text)

    # Load active messages from DB and build full context
    history = load_active_messages(conversation_id)

    messages_for_api = [{"role": "system", "text": BASE_SYSTEM_PROMPT}]
    messages_for_api.extend(history)

    try:
        reply_text, usage = call_yandex_llm(messages_for_api)
    except LlmError as e:
        print(f"\n[ERROR] LLM call failed: {e}")
        return

    # Save assistant reply
    save_message(conversation_id, "assistant", reply_text)

    # Stats and logging
    save_stats(conversation_id, usage)

    print("\nASSISTANT:\n" + reply_text)
    _print_usage(usage, messages_sent=len(messages_for_api), label="chat")


def summarize_conversation(conversation_id: int, max_tokens_for_summary: int) -> None:
    """
    Summarization:
    - get active messages
    - call LLM with summarization system prompt
    - archive previous non-summary messages
    - save summary as active system message
    - log token usage
    """
    history = load_active_messages(conversation_id)
    if not history:
        print("Nothing to summarize: no active messages.")
        return

    system_text = SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(
        max_tokens=max_tokens_for_summary
    )

    messages_for_api = [{"role": "system", "text": system_text}]
    messages_for_api.extend(history)

    try:
        reply_text, usage = call_yandex_llm(
            messages_for_api,
            temperature=0.3,
            max_tokens=max_tokens_for_summary,
        )
    except LlmError as e:
        print(f"\n[ERROR] LLM call failed: {e}")
        return

    # Archive all previous (non-summary) messages
    archive_non_summary_messages(conversation_id)

    # Save new summary message as active
    save_message(
        conversation_id,
        "system",
        reply_text,
        is_summary=True,
        is_active=True,
    )

    save_stats(conversation_id, usage)

    print("\nSUMMARY:\n" + reply_text)
    _print_usage(usage, messages_sent=len(messages_for_api), label="summary")
