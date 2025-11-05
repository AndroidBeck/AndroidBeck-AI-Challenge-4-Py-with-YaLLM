import os
import requests

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt-lite"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

payload = {
    "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
    "completionOptions": {
        "temperature": 0.6,
        "maxTokens": 1200,
        "stream": False
    },
    "messages": [
        {
            "role": "system",
            "text": "Ты ассистент дроид, способный помочь в галактических приключениях."
        },
        {
            "role": "user",
            "text": "Привет, Дроид! Мне нужна твоя помощь, чтобы узнать больше о Силе. Как я могу научиться ее использовать?"
        },
        {
            "role": "assistant",
            "text": "Привет! Чтобы овладеть Силой, тебе нужно понять ее природу. Сила находится вокруг нас и соединяет всю галактику. Начнем с основ медитации."
        }
    ]
}

headers = {
    "Authorization": f"Api-Key {YAC_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(YAGPT_URL, headers=headers, json=payload)
result = response.text
print(result)