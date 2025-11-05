import os
import sys
import json
import requests

# === Configuration ===
YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt-lite"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

if not YAC_FOLDER or not YAC_API_KEY:
    print("Missing env vars. Please set YAC_FOLDER and YAC_API_KEY.")
    sys.exit(1)

headers = {
    "Authorization": f"Api-Key {YAC_API_KEY}",
    "Content-Type": "application/json",
}

# === Define strict JSON schema ===
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"]
        },
    },
    "required": ["title", "description", "priority"],
    "additionalProperties": False,
}

# === Initialize chat ===
messages = [
    {
        "role": "system",
        "text": (
            "Ты ассистент-дроид, который отвечает только в JSON-формате "
            "по заданной схеме. Никогда не добавляй комментарии или текст "
            "вне JSON."
        ),
    }
]

def chat_completion(messages):
    """Send conversation and get structured JSON response."""
    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": 0.5,
            "maxTokens": 1200,
            "stream": False
        },
        "messages": messages,
        "jsonSchema": {"schema": schema},  # Enforce strict JSON
    }

    resp = requests.post(YAGPT_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        text = data["result"]["alternatives"][0]["message"]["text"]
        parsed = json.loads(text)
        return parsed
    except Exception:
        print("Unexpected response format:\n", json.dumps(data, ensure_ascii=False, indent=2))
        raise

def trim_history(messages, keep_last=20):
    """Limit conversation length to avoid large requests."""
    head = messages[:1] if messages and messages[0]["role"] == "system" else []
    tail = messages[len(messages)-2*keep_last:]
    return head + tail

# === Chat Loop ===
print("Chat (JSON mode) ready!")
print("Commands: /exit to quit, /reset to clear history.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "/exit":
        break
    if user_input.lower() == "/reset":
        messages = messages[:1]
        print("(history reset)\n")
        continue

    # 1) Add user message
    messages.append({"role": "user", "text": user_input})
    messages = trim_history(messages, keep_last=20)

    # 2) Get JSON response
    try:
        reply = chat_completion(messages)
    except Exception as e:
        print(f"Error: {e}\n")
        continue

    # 3) Add assistant message and display structured output
    messages.append({"role": "assistant", "text": json.dumps(reply, ensure_ascii=False)})
    print("\nAssistant JSON reply:")
    print(json.dumps(reply, ensure_ascii=False, indent=2))
    print()
