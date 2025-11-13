import os
import requests

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt"  # same as in previous days (change if needed)

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

SYSTEM_PROMPT = (
    "You are a helpful assistant participating in an AI learning challenge. "
    "Answer clearly and concisely."
)


def call_llm(messages):
    """Send messages to YandexGPT and return text + token usage + messages count."""

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": 0.6,
            "maxTokens": 1200,
            "stream": False,
        },
        "messages": messages,
    }

    headers = {
        "Authorization": f"Api-Key {YAC_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(YAGPT_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Text answer
    alt = data["result"]["alternatives"][0]["message"]
    answer_text = alt.get("text", "")

    # Token usage (strings in Yandex response, but we just print them)
    usage = data["result"].get("usage", {})
    input_tokens = usage.get("inputTextTokens", "0")
    completion_tokens = usage.get("completionTokens", "0")
    total_tokens = usage.get("totalTokens", "0")
    reasoning_tokens = usage.get("completionTokensDetails", {}).get(
        "reasoningTokens", "0"
    )

    # Number of messages we sent in THIS request
    messages_sent = len(payload["messages"])

    return (
        answer_text,
        input_tokens,
        completion_tokens,
        total_tokens,
        reasoning_tokens,
        messages_sent,
    )


def main():
    if not YAC_FOLDER or not YAC_API_KEY:
        print("Missing YAC_FOLDER or YAC_API_KEY environment variables.")
        return

    # Conversation history that we send to LLM each time
    # Always starts with system message
    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
    ]

    print("Day 8 – Dialog compression playground.")
    print("Type 'exit' to quit.")
    print("Use '/compress X' or '/summarize X' to compress history to ~X tokens.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Bye!")
            break

        # --- COMPRESSION / SUMMARIZATION COMMAND ---
        if user_input.startswith("/compress") or user_input.startswith("/summarize"):
            parts = user_input.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Use: /compress X or /summarize X, where X is a positive integer.")
                continue

            max_tokens = parts[1]

            # Build a special request: all previous messages + final instruction
            summarize_instruction = (
                f"Summarize the entire conversation so far (ignoring this instruction "
                f"itself) into a single concise assistant message, no longer than "
                f"{max_tokens} tokens. Preserve key facts, decisions, goals and "
                f"constraints. The output will be used as compressed memory for "
                f"next turns."
            )

            summary_messages = messages + [
                {"role": "user", "text": summarize_instruction}
            ]

            (
                summary_text,
                input_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens,
                messages_sent,
            ) = call_llm(summary_messages)

            print(f"\nAssistant (summary): {summary_text}\n")
            print(
                f"inputTextTokens = {input_tokens}, "
                f"completionTokens = {completion_tokens}, "
                f"totalTokens = {total_tokens}; "
                f"(reasoningTokens = {reasoning_tokens}); "
                f"messagesSent = {messages_sent}"
            )
            print()

            # 🔻 COMPRESSION STEP:
            # Clear history (except system) and keep only this one summary
            system_message = messages[0]  # keep original system prompt
            messages = [
                system_message,
                {"role": "assistant", "text": summary_text},
            ]

            continue

        # --- NORMAL CHAT TURN ---
        # Add user message to history
        messages.append({"role": "user", "text": user_input})

        (
            answer_text,
            input_tokens,
            completion_tokens,
            total_tokens,
            reasoning_tokens,
            messages_sent,
        ) = call_llm(messages)

        print(f"\nAssistant: {answer_text}\n")
        print(
            f"inputTextTokens = {input_tokens}, "
            f"completionTokens = {completion_tokens}, "
            f"totalTokens = {total_tokens}; "
            f"(reasoningTokens = {reasoning_tokens}); "
            f"messagesSent = {messages_sent}"
        )
        print()

        # Add assistant reply back to history
        messages.append({"role": "assistant", "text": answer_text})


if __name__ == "__main__":
    main()
