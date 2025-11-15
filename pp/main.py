# main.py
from db import (
    create_conversation,
    get_last_conversation_with_active_messages,
    init_db,
    deactivate_all_messages_for_conversation,
)
from chat_logic import ask_llm_for_chat_answer, summarize_conversation


def _select_or_create_conversation() -> int:
    """
    If there is a conversation with active messages, continue it.
    Otherwise, create a new one.
    """
    last_active = get_last_conversation_with_active_messages()

    if last_active is not None:
        conversation_id, title, created_at, updated_at = last_active
        print(
            f"Continuing conversation #{conversation_id}: '{title}' "
            f"(created_at={created_at}, updated_at={updated_at})"
        )
    else:
        conversation_id = create_conversation()
        print(f"Created new conversation #{conversation_id}")

    return conversation_id


def _print_banner() -> None:
    print("\nPersonal project")
    print("Type your question, or commands:")
    print("  /summarize X, /sum X, /compress X  – summarize active messages into ~X tokens (default 400)")
    print("  /new                               – start a NEW conversation (new topic)")
    print("  /exit                              – quit")


def main() -> None:
    init_db()

    conversation_id = _select_or_create_conversation()
    _print_banner()

    while True:
        user_input = input("\nYOU > ").strip()
        if not user_input:
            continue

        lower = user_input.lower()

        # Exit command
        if lower == "/exit":
            print("Goodbye!")
            break

        # Start a new conversation:
        # NOTE: keeping original behavior – just create a new conversation,
        # without deactivating messages of the previous one.
        if lower == "/new":
            conversation_id = create_conversation()
            print(f"Started NEW conversation #{conversation_id}")
            continue

        # Summarization commands: /summarize, /compress, /sum
        if (lower.startswith("/summarize")
                or lower.startswith("/compress")
                or lower.startswith("/sum")):
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
            continue

        # Deactivate all messages for current conversation
        if lower == "/deactivate":
            deactivate_all_messages_for_conversation(conversation_id)
            print(f"All messages in conversation #{conversation_id} were deactivated.")
            print("The next user message will start fresh context inside the same conversation.")
            continue

        # Normal chat turn
        ask_llm_for_chat_answer(conversation_id, user_input)


if __name__ == "__main__":
    main()
