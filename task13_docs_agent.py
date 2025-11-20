import asyncio
import datetime
import json
import os
from typing import Any, Dict

from mcp.client.stdio import StdioServerParameters
from mcp.client.session import ClientSession


# ---------- MCP CONNECTION ----------

async def connect_to_docs_server() -> ClientSession:
    """
    Connects to docs-tools MCP server via stdio.
    Adjust the command/path if your server filename is different.
    """
    params = StdioServerParameters(
        command="python",
        args=["task13_docs_tools_mcp_server.py"],
    )

    session = ClientSession(params)
    await session.__aenter__()  # manual "async with"
    return session


async def call_tool(
    session: ClientSession,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """
    Helper to call any MCP tool by name with JSON arguments.
    Returns parsed result.
    """
    tool_list = await session.list_tools()
    names = [t.name for t in tool_list.tools]
    if tool_name not in names:
        raise RuntimeError(f"Tool {tool_name!r} not found. Available: {names}")

    result = await session.call_tool(tool_name, arguments)

    # result.content is a list of Content objects (text/json)
    # We expect first item, JSON.
    if not result.content:
        return None

    item = result.content[0]
    if getattr(item, "type", "") == "json":
        return item.data
    else:
        # fallback: if it's text, just return text
        return getattr(item, "text", None)


# ---------- PIPELINE STEPS ----------

async def pipeline_search_summarize_save(query: str) -> None:
    session = await connect_to_docs_server()
    try:
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

        # Combine snippets into one text
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

    finally:
        await session.__aexit__(None, None, None)


# ---------- CLI LOOP ----------

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
