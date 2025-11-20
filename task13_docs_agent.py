import asyncio
import datetime
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


# ---------- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ВЫЗОВА ТУЛОВ ----------

async def call_tool_structured(
    session: ClientSession,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """
    Вызов MCP-тула и возврат structuredContent (dict / список / примитив).

    Если structuredContent нет, пытаемся достать текст из content.
    """
    # Сначала убеждаемся, что тул вообще есть
    tools = await session.list_tools()
    tool_names = [t.name for t in tools.tools]
    if tool_name not in tool_names:
        raise RuntimeError(f"Tool {tool_name!r} not found. Available: {tool_names}")

    result = await session.call_tool(tool_name, arguments)

    # 1) Предпочитаем structuredContent — для наших тулов с return type dict
    if hasattr(result, "structuredContent") and result.structuredContent is not None:
        return result.structuredContent

    # 2) Если structuredContent нет — пробуем разобрать content как текст / json
    if not result.content:
        return None

    # Ищем текстовый контент
    for content in result.content:
        if isinstance(content, types.TextContent):
            return content.text

    # Если вдруг json-подобное (EmbeddedResource и т.п.) — просто вернём «как есть»
    return [c for c in result.content]


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

    # stdio-клиент создаёт read/write потоки
    async with stdio_client(server_params) as (read, write):
        # ClientSession поверх этих потоков
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"\n=== Connecting pipeline for query: {query!r} ===")

            # 1) search_docs
            print("[1] Calling search_docs...")
            search_result = await call_tool_structured(
                session,
                "search_docs",
                {"query": query, "docs_dir": "docs"},
            )

            # search_docs возвращает dict {"matches": [...]}
            matches = (search_result or {}).get("matches", [])
            print(f"    Found {len(matches)} matches")

            if not matches:
                print("No matches found, nothing to summarize.")
                return

            # Склеиваем snippets в один текст
            combined_text = "\n\n".join(
                f"File: {m['path']}\nSnippet: {m['snippet']}"
                for m in matches
            )

            # 2) summarize_text
            print("[2] Calling summarize_text...")
            summarize_result = await call_tool_structured(
                session,
                "summarize_text",
                {
                    "text": combined_text,
                    "max_tokens": 300,
                },
            )

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
            save_result = await call_tool_structured(
                session,
                "save_to_file",
                {
                    "content": summary,
                    "filename": filename,
                    "directory": "summaries",
                    "mode": "append",
                },
            )

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

        # каждая команда — отдельный запуск пайплайна
        asyncio.run(pipeline_search_summarize_save(query))


if __name__ == "__main__":
    main()
