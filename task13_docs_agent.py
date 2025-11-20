import asyncio
import datetime
import json
from typing import Any, Dict, Tuple

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


# ---------- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ВЫЗОВА ТУЛОВ ----------

async def call_tool_json_or_text(
    session: ClientSession,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """
    Вызов MCP-тула и попытка вернуть dict (если текст – JSON),
    иначе обычную строку.
    """
    # Проверим, что тул существует
    tools = await session.list_tools()
    tool_names = [t.name for t in tools.tools]
    if tool_name not in tool_names:
        raise RuntimeError(f"Tool {tool_name!r} not found. Available: {tool_names}")

    result = await session.call_tool(tool_name, arguments)

    # Если вдруг есть structuredContent и он не None — вернём его
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent

    # Иначе забираем первый текстовый контент
    if not result.content:
        return None

    text_value = None

    for content in result.content:
        if isinstance(content, types.TextContent):
            text_value = content.text
            break

    if text_value is None:
        # нет текстового контента — вернём всё как есть (редкий случай)
        return [c for c in result.content]

    # Пытаемся распарсить как JSON
    text_value = text_value.strip()
    if text_value.startswith("{") or text_value.startswith("["):
        try:
            return json.loads(text_value)
        except Exception:
            # не получилось — вернём текст как есть
            return text_value

    # Не похоже на JSON — просто текст
    return text_value


# ---------- ОСНОВНОЙ PIPELINE ----------

async def pipeline_search_summarize_save(query: str) -> None:
    """
    Подключается к MCP-серверу и выполняет пайплайн:
    search_docs -> summarize_text -> save_to_file
    """
    server_params = StdioServerParameters(
        command="python",
        args=["task13_docs_tools_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"\n=== Connecting pipeline for query: {query!r} ===")

            # 1) search_docs
            print("[1] Calling search_docs...")
            search_result = await call_tool_json_or_text(
                session,
                "search_docs",
                {"query": query, "docs_dir": "docs"},
            )

            # search_result может быть dict ИЛИ текстом
            if isinstance(search_result, str):
                print("search_docs returned plain text, not JSON. Raw value:")
                print(search_result)
                print("Treating this as 'no matches'.")
                matches = []
            else:
                matches = (search_result or {}).get("matches", [])

            print(f"    Found {len(matches)} matches")

            if not matches:
                print("No matches found, nothing to summarize.")
                return

            combined_text = "\n\n".join(
                f"File: {m['path']}\nSnippet: {m['snippet']}"
                for m in matches
            )

            # 2) summarize_text
            print("[2] Calling summarize_text...")
            summarize_result = await call_tool_json_or_text(
                session,
                "summarize_text",
                {
                    "text": combined_text,
                    "max_tokens": 300,
                },
            )

            # summarize_result тоже может быть dict или строкой
            if isinstance(summarize_result, str):
                # допустим, тул выдал чистый текст — тогда это уже summary
                summary = summarize_result.strip()
            else:
                summary = (summarize_result or {}).get("summary", "").strip()

            if not summary:
                print("Empty summary returned, nothing to save.")
                return

            print("\n=== Summary ===")
            print(summary)

            # 3) save_to_file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.txt"

            print("\n[3] Calling save_to_file...")
            save_result = await call_tool_json_or_text(
                session,
                "save_to_file",
                {
                    "content": summary,
                    "filename": filename,
                    "directory": "summaries",
                    "mode": "append",
                },
            )

            if isinstance(save_result, str):
                print("save_to_file returned plain text:")
                print(save_result)
                saved_path = None
            else:
                saved_path = (save_result or {}).get("path")

            print(f"\nSummary saved to: {saved_path}")


# ---------- CLI ----------

def main():
    print("=== Day 13 – MCP tools composition (search → summarize → save) ===")
    print("Make sure 'task13_docs_tools_mcp_server.py' is available and that you have a 'docs/' directory with .txt/.md files.\n")

    while True:
        try:
            query = input("Enter your query (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query or query.lower() in ("exit", "quit"):
            print("Bye.")
            break

        asyncio.run(pipeline_search_summarize_save(query))


if __name__ == "__main__":
    main()
