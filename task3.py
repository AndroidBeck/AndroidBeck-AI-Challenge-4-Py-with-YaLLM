import os
import sys
import requests

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

if not YAC_FOLDER or not YAC_API_KEY:
    print("ERROR: Missing YAC_FOLDER or YAC_API_KEY environment variables.")
    sys.exit(1)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {YAC_API_KEY}",
}

SYSTEM_PROMPT = """
You are a proactive expert assistant.
Rules:
1. The user gives a theme (e.g., "create a mobile app").
2. You ask detailed questions to gather all necessary information.
3. Ask only one question at a time.
4. Maximum number of questions is 10.
5. When you have enough data to write the full answer, stop asking questions and clearly say something like:
   "I have enough information. Here is the final result:" and then provide it.
6. Be confident, structured, and concise. Never repeat questions.
7. Always continue the same conversation logically.
"""

def yagpt(messages):
    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {"temperature": 0.6, "maxTokens": 1200, "stream": False},
        "messages": messages,
    }
    resp = requests.post(YAGPT_URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["result"]["alternatives"][0]["message"]["text"]

def main():
    print("=== Proactive Assistant ===")
    theme = input("Enter the theme: ").strip()
    if not theme:
        print("Theme required.")
        return

    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": f"Theme: {theme}. Start asking your first questions."}
    ]

    while True:
        reply = yagpt(messages)
        print("\nAssistant:\n", reply)

        # detect completion by key phrases
        if any(kw in reply.lower() for kw in ["final result", "final version", "here is", "completed specification", "summary of everything", "final answer"]):
            print("\n--- End of conversation ---")
            break

        user_input = input("\nYour answer (press Enter to stop): ").strip()
        if not user_input:
            print("Conversation stopped.")
            break

        messages.append({"role": "assistant", "text": reply})
        messages.append({"role": "user", "text": user_input})

if __name__ == "__main__":
    main()
