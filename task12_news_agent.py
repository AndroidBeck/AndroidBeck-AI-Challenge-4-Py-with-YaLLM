# task12_news_agent.py

# How to run:
# pip install mcp httpx requests
# set YAC_FOLDER=...
# set YAC_API_KEY=...
# python task12_news_agent.py

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt"  # or your preferred model

# Path to your MCP server file
NEWS_WEATHER_SERVER_PATH = "news_weather_mcp_server.py"

# How often to generate summary (minutes)
SUMMARY_INTERVAL_MINUTES = 30

# Where to store summaries (our "reminder log")
SUMMARY_LOG_PATH = "news_weather_summaries.jsonl"


def get_env_or_die(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} environment variable")
    return value


YAC_FOLDER = get_env_or_die("YAC_FOLDER")
YAC_API_KEY = get_env_or_die("YAC_API_KEY")


def call_yandex(prompt: str, temperature: float = 0.3, max_tokens: int = 800):
    """
    Simple wrapper to call YandexGPT with a single system+user prompt.
    Returns (answer_text, token_info_dict).
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

    # --- Extract text ---
    try:
        alternatives = data["result"]["alternatives"]
        message = alternatives[0]["message"]
        answer_text = message.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(
            f"Unexpected response format when reading text: {e}\n"
            f"Raw: {json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    # --- Extract usage / tokens ---
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


async def call_news_weather_tool_via_mcp(
    session: ClientSession,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call MCP tool 'get_news_and_weather' and return a plain dict.
    """
    result = await session.call_tool("get_news_and_weather", arguments=arguments)

    # Preferred: structuredContent (dict)
    if getattr(result, "structuredContent", None):
        return result.structuredContent  # type: ignore[return-value]

    # Fallback: parse text content as JSON if possible
    for content in result.content:
        if isinstance(content, types.TextContent):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                return {"raw": content.text}

    return {"raw": "No content returned from news+weather tool."}


def build_summary_prompt(data: Dict[str, Any]) -> str:
    """
    Build a prompt for YandexGPT to summarize news+weather JSON.
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
5. В конце можно добавить одну строку-комментарий вроде "В целом день спокойный" или "Много новостей про технологии".

Вот реальные данные в формате JSON:

{json.dumps(data, ensure_ascii=False, indent=2)}
""".strip()


def append_summary_to_file(
    when: datetime,
    raw_data: Dict[str, Any],
    summary: str,
    tokens: Dict[str, Any],
    path: str = SUMMARY_LOG_PATH,
) -> None:
    """
    Append one summary record to JSONL file.
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


async def run_news_weather_agent(session: ClientSession):
    """
    Infinite loop:
    - every N minutes:
      - call MCP tool get_news_and_weather
      - send JSON to Yandex for summarization
      - print and log summary
    """
    interval_seconds = SUMMARY_INTERVAL_MINUTES * 60

    print("=== Day 12 – News + Weather Agent (YandexGPT + MCP) ===")
    print(f"Summary interval: {SUMMARY_INTERVAL_MINUTES} minutes")
    print("Press Ctrl+C to stop.\n")

    while True:
        started_at = datetime.now()
        print(
            f"[{started_at.isoformat()}] Fetching news+weather via MCP tool...",
            flush=True,
        )

        # 1) Call MCP tool
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
            print(f"[Error while calling news_weather MCP tool] {e}", flush=True)
            await asyncio.sleep(interval_seconds)
            continue

        print(
            "[News+Weather raw JSON]:",
            json.dumps(data, ensure_ascii=False, indent=2),
            sep="\n",
            flush=True,
        )

        # If tool returned error, just log and wait
        if "error" in data:
            print(f"[Tool error] {data['error']}", flush=True)
            await asyncio.sleep(interval_seconds)
            continue

        # 2) Build prompt for Yandex summarization
        prompt = build_summary_prompt(data)

        # 3) Call YandexGPT
        try:
            summary_text, token_info = call_yandex(prompt)
        except Exception as e:
            print(f"[Error while calling YandexGPT] {e}", flush=True)
            await asyncio.sleep(interval_seconds)
            continue

        # 4) Print summary
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
        print("=" * 60, flush=True)

        # 5) Save summary to JSONL file (our "reminder log")
        append_summary_to_file(started_at, data, summary_text, token_info)

        # 6) Wait until next tick
        await asyncio.sleep(interval_seconds)


async def main_async():
    """
    Connect to the news+weather MCP server over stdio and run the agent loop.
    This will spawn `python news_weather_mcp_server.py` as a child process.
    """
    server_params = StdioServerParameters(
        command="python",
        args=[NEWS_WEATHER_SERVER_PATH],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # (Optional) list tools once to sanity-check
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Connected to MCP news+weather server. Tools: {tool_names}\n")

            await run_news_weather_agent(session)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except RuntimeError as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        sys.exit(0)
