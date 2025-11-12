import os
import sys
import json
import requests

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt"  # or your preferred model


def get_env_or_die(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} environment variable")
    return value


YAC_FOLDER = get_env_or_die("YAC_FOLDER")
YAC_API_KEY = get_env_or_die("YAC_API_KEY")


def call_yandex(prompt: str, temperature: float = 0.6, max_tokens: int = 1500):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
    }

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "stream": False,
        },
        "messages": [
            {
                "role": "system",
                "text": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "text": prompt,
            },
        ],
    }

    response = requests.post(YAGPT_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    # --- Extract text ---
    try:
        alternatives = data["result"]["alternatives"]
        message = alternatives[0]["message"]
        answer_text = message.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected response format when reading text: {e}\nRaw: {json.dumps(data, ensure_ascii=False, indent=2)}")

    # --- Extract usage / tokens ---
    usage = data.get("result", {}).get("usage", {}) or {}
    completion_details = usage.get("completionTokensDetails", {}) or {}

    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")
    reasoning_tokens = completion_details.get("reasoningTokens")

    # They sometimes come as strings; keep them as-is or convert if you prefer:
    # input_tokens = int(input_tokens) if input_tokens is not None else None

    token_info = {
        "inputTextTokens": input_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": total_tokens,
        "reasoningTokens": reasoning_tokens,
    }

    return answer_text, token_info


def main():
    print("=== Day 7 – YandexGPT Token Counter ===")
    print("Type your prompt and press Enter.")
    print("Type 'exit', 'quit' or 'q' to stop.\n")

    while True:
        try:
            user_prompt = input("Your prompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_prompt.strip().lower() in {"exit", "quit", "q"}:
            print("Exiting. Bye!")
            break

        if not user_prompt.strip():
            continue  # ignore empty lines

        try:
            answer, tokens = call_yandex(user_prompt)
        except Exception as e:
            print(f"\n[Error while calling YandexGPT]\n{e}\n")
            continue

        print("\nAssistant:")
        print(answer)
        print("\nTokens:")
        print(
            f"inputTextTokens = {tokens.get('inputTextTokens')}, "
            f"completionTokens = {tokens.get('completionTokens')}, "
            f"totalTokens = {tokens.get('totalTokens')}; "
            f"(reasoningTokens = {tokens.get('reasoningTokens')})"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
