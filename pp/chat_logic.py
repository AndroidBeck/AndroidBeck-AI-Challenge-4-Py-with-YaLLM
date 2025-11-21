# chat_logic.py
from typing import Any, Dict, Tuple, Optional

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
# CORE OPERATIONS (для UI и CLI)
# =========================

def chat_turn(
    conversation_id: int,
    user_text: str,
    model_name: str,
) -> Tuple[Optional[str], Dict[str, Any], int]:
    """
    Выполняет один шаг чата и ВОЗВРАЩАЕТ результат, не печатая ничего.
    Используется как из UI, так и из CLI-обёртки.

    Возвращает:
      (reply_text | None при ошибке, usage_dict (может быть пустым), messages_sent)
    """
    # Сохраняем пользовательское сообщение
    save_message(conversation_id, "user", user_text)

    # Загружаем активные сообщения и строим контекст
    history = load_active_messages(conversation_id)

    messages_for_api = [{"role": "system", "text": BASE_SYSTEM_PROMPT}]
    messages_for_api.extend(history)

    try:
        reply_text, usage = call_yandex_llm(
            messages_for_api,
            temperature=0.6,
            max_tokens=1500,
            model_name=model_name,
        )
    except LlmError as e:
        # Логируем текст ошибки в консоль (видно и при запуске GUI из консоли)
        print(f"[LLM ERROR in chat_turn] {e}")
        return None, {}, len(messages_for_api)

    # Сохраняем ответ ассистента
    save_message(conversation_id, "assistant", reply_text)

    # Сохраняем статистику
    save_stats(conversation_id, usage, model_name=model_name)

    return reply_text, usage, len(messages_for_api)


def summarize_conversation_core(
    conversation_id: int,
    max_tokens_for_summary: int,
    model_name: str,
) -> Tuple[Optional[str], Dict[str, Any], int]:
    """
    Ядро summarization-логики, без печати, подходит для UI.
    Делает:
      - берёт активные сообщения
      - вызывает LLM с summarization prompt
      - архивирует предыдущие (не summary) сообщения
      - сохраняет новый summary как активный system-message
      - сохраняет статистику

    Возвращает:
      (summary_text | None если нечего суммировать или ошибка,
       usage_dict (может быть пустым),
       messages_sent)

    При ошибке LLM или отсутствии истории БД не модифицируется (кроме уже существующего состояния).
    """
    history = load_active_messages(conversation_id)
    if not history:
        # Нечего суммировать
        return None, {}, 0

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
            model_name=model_name,
        )
    except LlmError as e:
        print(f"[LLM ERROR in summarize_conversation_core] {e}")
        return None, {}, len(messages_for_api)

    # Архивируем все предыдущие не-summary сообщения
    archive_non_summary_messages(conversation_id)

    # Сохраняем новый summary как активное system-сообщение
    save_message(
        conversation_id,
        "system",
        reply_text,
        is_summary=True,
        is_active=True,
    )

    # Сохраняем статистику
    save_stats(conversation_id, usage, model_name=model_name)

    return reply_text, usage, len(messages_for_api)


# =========================
# CLI-ОБЁРТКИ (оставляем старый интерфейс)
# =========================

def ask_llm_for_chat_answer(
    conversation_id: int,
    user_text: str,
    model_name: str,
) -> None:
    """
    Обёртка для консольной версии:
    - вызывает chat_turn
    - печатает результат и usage
    """
    reply_text, usage, messages_sent = chat_turn(
        conversation_id, user_text, model_name
    )

    if reply_text is None:
        print(f"\n[ERROR] LLM call failed.")
        return

    print("\nASSISTANT:\n" + reply_text)
    _print_usage(usage, messages_sent=messages_sent, label="chat")


def summarize_conversation(
    conversation_id: int,
    max_tokens_for_summary: int,
    model_name: str,
) -> None:
    """
    Обёртка для консольной версии summarization:
    - вызывает summarize_conversation_core
    - печатает summary и usage
    """
    summary_text, usage, messages_sent = summarize_conversation_core(
        conversation_id,
        max_tokens_for_summary,
        model_name,
    )

    if messages_sent == 0:
        print("Nothing to summarize: no active messages.")
        return

    if summary_text is None:
        print("\n[ERROR] Summarization failed.")
        return

    print("\nSUMMARY:\n" + summary_text)
    _print_usage(usage, messages_sent=messages_sent, label="summary")
