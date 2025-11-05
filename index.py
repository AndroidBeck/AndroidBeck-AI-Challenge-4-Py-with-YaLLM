import os
import sys
import json
import requests

# --- Config ---
YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt-lite"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

if not YAC_FOLDER or not YAC_API_KEY:
    print("Missing env vars. Please set YAC_FOLDER and YAC_API_KEY.")
    sys.exit(1)

headers = {
    "Authorization": f"Api-Key {YAC_API_KEY}",
    "Content-Type": "application/json"
}

# Start the conversation with an optional system prompt
messages = [
    {
        "role": "system",
        "text": "Ты ассистент дроид, способный помочь в галактических приключениях."
    }
]

def chat_completion(messages):
    """Send the conversation to YandexGPT and return assistant text."""
    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": 0.6,
            "maxTokens": 1200,
            "stream": False
        },
        "messages": messages
    }

    resp = requests.post(YAGPT_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    # Expected response shape (typical for YandexGPT):
    # {
    #   "result": {
    #     "alternatives": [
    #       {
    #         "message": {"role": "assistant", "text": "..."},
    #         "status": "ALTERNATIVE_STATUS_FINAL"
    #       }
    #     ],
    #     "usage": {...}
    #   }
    # }
    try:
        return data["result"]["alternatives"][0]["message"]["text"]
    except Exception:
        # If schema changes, show what we got for quick debugging
        raise RuntimeError("Unexpected response format:\n" + json.dumps(data, ensure_ascii=False, indent=2))


def trim_history(messages, keep_last=20):
    """Limit history to avoid huge requests (keeps system + last N pairs)."""
    # Keep the first message if it's a system prompt
    head = messages[:1] if messages and messages[0]["role"] == "system" else []
    tail = messages[len(messages)-2*keep_last:]
    return head + tail


print("Chat ready. Type your message. Commands: /exit to quit, /reset to clear history.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "/exit":
        break
    if user_input.lower() == "/reset":
        messages = messages[:1]  # keep only system
        print("(history reset)\n")
        continue

    # 1) add user message
    messages.append({"role": "user", "text": user_input})

    # 2) (optional) trim to keep the prompt small
    messages = trim_history(messages, keep_last=20)

    # 3) call the model
    try:
        assistant_text = chat_completion(messages)
    except Exception as e:
        print(f"Error: {e}\n")
        continue

    # 4) add assistant reply to history and print it
    messages.append({"role": "assistant", "text": assistant_text})
    print(f"Assistant: {assistant_text}\n")
