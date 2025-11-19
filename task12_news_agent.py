# task12_news_agent.py  (обновлённая простая версия)

# How to run:
# pip install mcp httpx requests
# set YAC_FOLDER=...
# set YAC_API_KEY=...

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Any, Dict

import requests

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt"  # или другой, если хочешь

# Путь к MCP-серверу
NEWS_WEATHER_SERVER_PATH = "news_weather_mcp_server.py"

# Лог с саммари (JSONL)
SUMMARY_LOG_PATH = "news_weather_summaries.jsonl"


# ----------------- ENV / Yandex -----------------


def get_env_or_die(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} environment variable")
    return value


YAC_FOLDER = get_env_or_die("YAC_FOLDER")
YAC_API_KEY = get_env_or_die("YAC_API_KEY")


def call_yandex(prompt: str, temperature: float = 0.3, max_tokens: int = 800):
    """
    Обёртка над YandexGPT: один system+user промпт.
    Возвращает (текст_ответа, словарь_с_токенами).
    """
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
                "text": (
                    "Ты — ассистент, который делает краткие, понятные сводки "
                    "погоды и новостей для человека. Пиши по-русски."
                ),
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

    # Достаём текст
    try:
        alternatives = data["result"]["alternatives"]
        message = alternatives[0]["message"]
        answer_text = message.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(
            f"Unexpected response format when reading text: {e}\n"
            f"Raw: {json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    # Токены
    usage = data.get("result", {}).get("usage", {}) or {}
    completion_details = usage.get("completionTokensDetails", {}) or {}

    input_tokens = usage.get("inputTextTokens")
    completion_tokens = usage.get("completionTokens")
    total_tokens = usage.get("totalTokens")
    reasoning_tokens = completion_details.get("reasoningTokens")

    token_info = {
        "inputTextTokens": input_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": total_tokens,
        "reasoningTokens": reasoning_tokens,
    }

    return answer_text, token_info


# ----------------- MCP tool call -----------------


async def call_news_weather_tool_via_mcp(
    session: ClientSession,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Вызов MCP-инструмента get_news_and_weather, возврат dict.
    """
    result = await session.call_tool("get_news_and_weather", arguments=arguments)

    # Если FastMCP вернул structuredContent — берём его
    if getattr(result, "structuredContent", None):
        return result.structuredContent  # type: ignore[return-value]

    # Иначе пытаемся взять текст и распарсить JSON
    for content in result.content:
        if isinstance(content, types.TextContent):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                return {"raw": content.text}

    return {"raw": "No content returned from news+weather tool."}


# ----------------- Prompt для саммари -----------------


def build_summary_prompt(data: Dict[str, Any]) -> str:
    """
    Собираем промпт для YandexGPT на основе JSON от MCP.
    """
    return f"""
Ты — агент, который делает периодическую сводку погоды и новостей.

Тебе даётся JSON с такими полями (примерная структура):
- location: информация о городе
- units: единицы измерения
- current_weather: температура, ветер, код погоды
- news: список новостей (title, url, points, author, created_at)
- fetched_at: время получения данных

ТВОЯ ЗАДАЧА:
1. Кратко описать текущую погоду (1–2 предложения).
2. Сформировать дайджест новостей: выделить 3–5 самых интересных заголовков.
3. Не придумывать факты, использовать только то, что есть в JSON.
4. Писать по-русски, коротко и по делу.
5. В конце можно добавить одну строку-комментарий вроде "В целом день спокойный"
   или "Много новостей про технологии".

Вот реальные данные в формате JSON:

{json.dumps(data, ensure_ascii=False, indent=2)}
""".strip()


# ----------------- Логирование саммари -----------------


def append_summary_to_file(
    when: datetime,
    raw_data: Dict[str, Any],
    summary: str,
    tokens: Dict[str, Any],
    path: str = SUMMARY_LOG_PATH,
) -> None:
    """
    Добавляем одну запись в JSONL-файл: raw данные + саммари + токены.
    """
    record = {
        "generated_at": when.isoformat(),
        "raw_data": raw_data,
        "summary": summary,
        "tokens": tokens,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


# ----------------- Основной агент (фиксированный интервал) -----------------


async def run_news_weather_agent_fixed_interval(
    session: ClientSession,
    interval_seconds: int,
):
    """
    Бесконечный цикл:
      - каждые interval_seconds:
        - вызываем get_news_and_weather через MCP
        - отправляем JSON в Yandex для саммари
        - печатаем и логируем
    """
    print("=== Day 12 – News + Weather Agent (YandexGPT + MCP) ===")
    print(
        f"Summary interval: {interval_seconds} seconds "
        f"({interval_seconds // 60} minutes)"
    )
    print("Press Ctrl+C to stop.\n")

    while True:
        started_at = datetime.now()
        print(
            f"[{started_at.isoformat()}] Fetching news+weather via MCP tool...",
            flush=True,
        )

        # 1) MCP tool
        try:
            data = await call_news_weather_tool_via_mcp(
                session,
                {
                    "location": "Amsterdam",
                    "country_code": "NL",
                    "units": "metric",
                    "news_limit": 5,
                },
            )
        except Exception as e:
            print(f"[Error calling MCP tool] {e}", flush=True)
            await asyncio.sleep(interval_seconds)
            continue

        print(
            "[News+Weather raw JSON]:",
            json.dumps(data, ensure_ascii=False, indent=2),
            sep="\n",
            flush=True,
        )

        if "error" in data:
            print(f"[Tool error] {data['error']}", flush=True)
            await asyncio.sleep(interval_seconds)
            continue

        # 2) Промпт для Яндекса
        prompt = build_summary_prompt(data)

        # 3) Вызов YandexGPT
        try:
            summary_text, token_info = call_yandex(prompt)
        except Exception as e:
            print(f"[Yandex error] {e}", flush=True)
            await asyncio.sleep(interval_seconds)
            continue

        # 4) Вывод
        print("\n=== SUMMARY ===")
        print(summary_text)
        print("\nTokens:")
        print(
            f"  inputTextTokens = {token_info.get('inputTextTokens')}, "
            f"completionTokens = {token_info.get('completionTokens')}, "
            f"totalTokens = {token_info.get('totalTokens')}; "
            f"(reasoningTokens = {token_info.get('reasoningTokens')})",
            flush=True,
        )
        print("=" * 60)

        # 5) Логируем
        append_summary_to_file(started_at, data, summary_text, token_info)

        # 6) Ждём следующий тик
        await asyncio.sleep(interval_seconds)


# ----------------- main_async: спрашиваем интервал и запускаем MCP-клиент -----------------


async def main_async():
    # 1. Спрашиваем интервал у пользователя
    while True:
        try:
            raw = input("Введите интервал обновления (в секундах): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Exit.")
            return

        if not raw:
            continue

        try:
            interval_seconds = int(raw)
            if interval_seconds <= 0:
                print("Интервал должен быть больше 0.")
                continue
            break
        except ValueError:
            print("Введите число, пожалуйста.")

    # 2. Стартуем MCP-сервер как дочерний процесс (stdio)
    server_params = StdioServerParameters(
        command="python",
        args=[NEWS_WEATHER_SERVER_PATH],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Проверим список тулов
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Connected to MCP news+weather server. Tools: {tool_names}\n")

            # Запускаем агента с фиксированным интервалом
            await run_news_weather_agent_fixed_interval(session, interval_seconds)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
