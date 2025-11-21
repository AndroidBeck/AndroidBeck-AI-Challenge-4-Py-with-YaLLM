# task14_orchestrator_agent.py
"""
Day 14 – Orchestrator agent:
Uses TWO MCP servers, each with multiple tools.

Servers:
  - task13_docs_tools_mcp_server.py
      * search_docs(query: str)
      * summarize_text(text: str, ...)
      * save_to_file(content: str, filename_hint: str)

  - task14_news_weather_mcp_server.py
      * get_news(topic: str, ...)
      * get_weather(location: str, ...)

Flow:
  1) Ask user for research topic and (optionally) city.
  2) search_docs on topic → summarize_text → "Background".
  3) get_news on topic → "Latest news".
  4) get_weather on city → "Context" (optional).
  5) Call YandexGPT to combine into a structured report.
  6) save_to_file final report via docs MCP.
"""

import asyncio
import datetime
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ---- load .env FIRST ----
load_dotenv()  # will load YAC_FOLDER and YAC_API_KEY from .env

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -----------------------------
# YandexGPT config
# -----------------------------

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")


def call_yandex_report_model(prompt: str, model: str = "yandexgpt") -> str:
    """
    Call YandexGPT to generate a structured research report.
    """
    if not YAC_FOLDER or not YAC_API_KEY:
        return (
            "Error: YandexGPT config is missing. "
            "Please ensure .env contains YAC_FOLDER and YAC_API_KEY."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
    }

    body = {
        "modelUri": f"gpt://{YAC_FOLDER}/{model}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 1500,
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "You are an analyst creating concise, structured research reports. "
                    "Write clear sections, avoid markdown unless asked, "
                    "and keep a professional tone."
                ),
            },
            {
                "role": "user",
                "text": prompt,
            },
        ],
    }

    resp = requests.post(YAGPT_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["result"]["alternatives"][0]["message"]["text"]
    except Exception:
        return f"Error parsing YandexGPT response: {json.dumps(data, ensure_ascii=False)[:1000]}"


# -----------------------------
# MCP client helpers
# -----------------------------

async def call_tool(
    session: ClientSession,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """
    Generic tool call wrapper with basic error handling.
    """
    result = await session.call_tool(tool_name, arguments)

    if not result.content:
        return None

    item = result.content[0]

    if item.type == "text":
        text = item.text
        try:
            return json.loads(text)
        except Exception:
            return text

    if item.type == "json":
        return item.json

    return [c.to_dict() for c in result.content]


# -----------------------------
# Docs MCP: helpers
# -----------------------------

async def docs_search_and_summarize(
    docs_session: ClientSession,
    topic: str,
    max_docs: int = 3,
) -> str:
    """
    Use docs MCP to search local docs and summarize them into a "background" text.
    """
    print(f"[Docs] Searching docs for topic: {topic!r} ...")
    search_res = await call_tool(docs_session, "search_docs", {"query": topic})

    if not isinstance(search_res, dict):
        return f"Could not parse search_docs result: {search_res!r}"

    matches = search_res.get("matches") or []
    print(f"[Docs] Found {len(matches)} matches")

    if not matches:
        return f"No local documents found for topic: {topic}."

    texts: List[str] = []
    for m in matches[:max_docs]:
        content = m.get("content") or ""
        if content:
            texts.append(content)

    if not texts:
        return f"No content in matched documents for topic: {topic}."

    combined_text = "\n\n---\n\n".join(texts)

    print("[Docs] Calling summarize_text on collected content...")
    summary_res = await call_tool(docs_session, "summarize_text", {"text": combined_text})

    if isinstance(summary_res, dict) and "summary" in summary_res:
        return summary_res["summary"]
    if isinstance(summary_res, str):
        return summary_res

    return f"Could not parse summarize_text result: {summary_res!r}"

async def docs_save_report(
    docs_session: ClientSession,
    report_text: str,
    topic: str,
) -> Optional[str]:
    """
    Save final report.

    We do it in two steps:
    1) Agent itself writes to summaries/summary_YYYYMMDD_HHMMSS.txt
       (like in previous days).
    2) Optionally call docs MCP save_to_file just to keep orchestration,
       but we ignore its returned path.

    The path we RETURN (and print) is the timestamped file created by the agent.
    """
    # 1) Agent-side file write with timestamped filename
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summaries_dir = os.path.join(base_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}.txt"
    local_path = os.path.join(summaries_dir, filename)

    print(f"[Docs] Writing report locally to: {local_path!r}")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 2) Call docs MCP save_to_file as part of orchestration (optional, best-effort)
    try:
        print(f"[Docs] Also calling save_to_file via MCP (filename_hint={filename!r}) ...")
        _ = await call_tool(
            docs_session,
            "save_to_file",
            {"content": report_text, "filename_hint": filename},
        )
    except Exception as e:
        # Log but do not fail the whole pipeline
        print(f"[Docs] Warning: save_to_file via MCP failed: {e!r}")

    # We ALWAYS return the agent-created local_path
    return local_path


# -----------------------------
# News + Weather MCP: helpers
# -----------------------------

async def news_get_news(
    news_session: ClientSession,
    topic: str,
    language: str = "en",
    page_size: int = 5,
) -> Dict[str, Any]:
    print(f"[News] Fetching news for topic: {topic!r} ...")
    res = await call_tool(
        news_session,
        "get_news",
        {"topic": topic, "language": language, "page_size": page_size},
    )
    if isinstance(res, dict):
        return res
    return {"raw": res}


async def news_get_weather(
    news_session: ClientSession,
    city: str,
    units: str = "metric",
) -> Dict[str, Any]:
    print(f"[Weather] Fetching weather for city: {city!r} ...")
    res = await call_tool(
        news_session,
        "get_weather",
        {"location": city, "units": units},
    )
    if isinstance(res, dict):
        return res
    return {"raw": res}


# -----------------------------
# Orchestration logic
# -----------------------------

async def orchestrate(topic: str, city: str) -> None:
    """
    Main Day 14 orchestration pipeline.
    """
    print("\n=== Day 14 – Orchestration (Docs + News + Weather) ===")

    docs_params = StdioServerParameters(
        command=sys.executable,
        args=["task13_docs_tools_mcp_server.py"],
    )
    news_params = StdioServerParameters(
        command=sys.executable,
        args=["task14_news_weather_mcp_server.py"],
    )

    # NESTED contexts to avoid cancel-scope issues:
    async with stdio_client(docs_params) as (docs_r, docs_w):
        async with ClientSession(docs_r, docs_w) as docs_session:
            await docs_session.initialize()

            async with stdio_client(news_params) as (news_r, news_w):
                async with ClientSession(news_r, news_w) as news_session:
                    await news_session.initialize()

                    # 1) Docs: background
                    background_text = await docs_search_and_summarize(docs_session, topic)

                    # 2) News: latest headlines
                    news_data = await news_get_news(news_session, topic, language="en", page_size=5)
                    articles = news_data.get("articles") or []
                    news_section_lines: List[str] = []
                    if articles:
                        for idx, art in enumerate(articles, start=1):
                            line = f"{idx}. {art.get('title') or 'No title'}"
                            desc = art.get("description") or ""
                            if desc:
                                line += f"\n   {desc}"
                            source = art.get("source") or ""
                            published = art.get("published_at") or ""
                            if source or published:
                                line += f"\n   [{source} | {published}]"
                            news_section_lines.append(line)
                    else:
                        err = news_data.get("error")
                        if err:
                            news_section_lines.append(f"Error fetching news: {err}")
                        else:
                            news_section_lines.append("No news articles found or parsed.")

                    news_section_text = "\n\n".join(news_section_lines)

                    # 3) Weather: optional
                    weather_section_text = ""
                    if city:
                        weather_data = await news_get_weather(news_session, city, units="metric")
                        if "error" in weather_data:
                            weather_section_text = f"Error fetching weather for {city}: {weather_data['error']}"
                        else:
                            cw = (weather_data.get("current_weather") or {})
                            loc = (weather_data.get("location") or {})
                            note = (loc.get("note") or "")
                            weather_section_text = (
                                f"Location: {loc.get('resolved_name')}, {loc.get('country')} "
                                f"({loc.get('latitude')}, {loc.get('longitude')})\n"
                                f"Temperature: {cw.get('temperature')}°C\n"
                                f"Wind: {cw.get('windspeed')} km/h, direction {cw.get('winddirection')}\n"
                                f"Time: {cw.get('time')}"
                            )
                            if note:
                                weather_section_text += f"\nNOTE: {note}"

                    # 4) Build YandexGPT prompt
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    user_prompt = textwrap.dedent(
                        f"""
                        Create a structured report about the topic: "{topic}".

                        Current date/time: {now_str}

                        SECTION 1. Background (from my local documents)
                        ----------------------------------------------
                        {background_text}

                        SECTION 2. Latest news on this topic
                        ------------------------------------
                        Raw news data (already partially formatted):

                        {news_section_text}

                        SECTION 3. Context (weather) – optional
                        ---------------------------------------
                        {weather_section_text or '(no weather data requested or available)'}

                        Requirements:
                        - Write a clear, concise report with 3–5 sections.
                        - Start with a short summary (3–5 bullet points).
                        - Then have separate sections for background, recent developments, and context.
                        - Do NOT mention that this comes from tools or MCP – just present it as a normal report.
                        - Limit length to a few paragraphs per section.
                        """
                    ).strip()

                    print("\n[LLM] Calling YandexGPT to generate final report...")
                    final_report = call_yandex_report_model(user_prompt)

                    print("\n=== Final Report (preview) ===")
                    print(final_report[:1000])
                    if len(final_report) > 1000:
                        print("\n... (truncated, full text will be saved to file)")

                    # 5) Save via docs MCP
                    saved_path = await docs_save_report(docs_session, final_report, topic)

                    print("\n=== Saved Report ===")
                    if saved_path:
                        print(f"Report saved to: {saved_path}")
                    else:
                        print("Could not determine saved file path from save_to_file response.")


def main() -> None:
    print("=== Day 14 – Orchestration agent ===")
    print("This script will combine:")
    print("- Local docs (search + summarize)")
    print("- Fresh news about your topic")
    print("- Optional weather for a city")
    print()

    topic = input("Enter research topic (or 'exit' to quit): ").strip()
    if not topic or topic.lower() == "exit":
        print("Exiting.")
        return

    city = input("Enter city for weather (leave blank to skip): ").strip()

    asyncio.run(orchestrate(topic, city))


if __name__ == "__main__":
    main()
