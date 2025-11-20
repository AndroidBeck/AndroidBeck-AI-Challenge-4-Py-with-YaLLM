import asyncio
import datetime
from typing import Any, Dict

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession


# ---------- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ВЫЗОВА ТУЛОВ ----------

async def call_tool(
    session: ClientSession,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """
    Вызов MCP-тула по имени с JSON-аргументами.
    Возвращает распарсенный результат (dict или строку).
    """
    tool_list = await session.list_tools()
    names = [t.name for t in tool_list.tools]
    if tool_name not in names:
        raise RuntimeError(f"Tool {tool_name!r} not found. Available: {names}")

    result = await session.call_tool(tool_name, arguments)

    if not result.content:
        return None

    item = result.content[0]

    # JsonContent обычно имеет .data
    if hasattr(item, "data"):
        return item.data
    # TextContent — .text
    if hasattr(item, "text"):
        return item.text

    return None


# ---------- ОСНОВНОЙ PIPELINE ----------

async def pipeline_search_summarize_save(query: str) -> None:
    """
    Открывает MCP-сервер через stdio и выполняет pipeline:
    search_docs -> summarize_text -> save_to_file
    """
    params = StdioServerParameters(
        command="python",
        args=["task13_docs_tools_mcp_server.py"],  # имя твоего сервера
    )

    # Открываем stdio-клиент и сессию
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"\n=== Connecting pipeline for query: {query!r} ===")

            # 1) search_docs
            print("[1] Calling search_docs...")
            search_result = await call_tool(
                session,
                "search_docs",
                {"query": query, "docs_dir": "docs"},
            )
            matches = (search_result or {}).get("matches", [])
            print(f"    Found {len(matches)} matches")

            if not matches:
                print("No matches found, nothing to summarize.")
                return

            # Склеиваем сниппеты в один текст
            combined_text = "\n\n".join(
                f"File: {m['path']}\nSnippet: {m['snippet']}"
                for m in matches
            )

            # 2) summarize_text
            print("[2] Calling summarize_text...")
            summarize_result = await call_tool(
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
            save_result = await call_tool(
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

        asyncio.run(pipeline_search_summarize_save(query))


if __name__ == "__main__":
    main()
