import os
import json
import glob
import textwrap
import requests

from mcp.server.fastmcp import FastMCPServer

# To run this:
# pip install fastmcp mcp-client
# pip install requests
# Put at least one file into docs/, e.g. docs/doc1.txt
# python task13_docs_agent.py


# =========================
# YandexGPT CONFIG
# =========================

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

if not YAC_FOLDER:
    raise RuntimeError("YAC_FOLDER env var is not set")
if not YAC_API_KEY:
    raise RuntimeError("YAC_API_KEY env var is not set")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {YAC_API_KEY}",
}

# =========================
# MCP SERVER
# =========================

server = FastMCPServer("docs-tools")


def _load_docs_text(docs_dir: str) -> list[tuple[str, str]]:
    """Return list of (path, text) for all .txt/.md files in docs_dir."""
    patterns = ["*.txt", "*.md"]
    results: list[tuple[str, str]] = []

    for pattern in patterns:
        for path in glob.glob(os.path.join(docs_dir, pattern)):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                results.append((path, text))
            except Exception as e:
                print(f"Failed to read {path}: {e}")

    return results


def _call_yandex_summarize(text: str, max_tokens: int = 300) -> str:
    prompt = textwrap.dedent(
        f"""
        You are a helpful assistant that summarizes documents.

        Summarize the following text in a concise form, preserving key facts:

        ---
        {text}
        ---
        """
    )

    body = {
        "modelUri": f"gpt://{YAC_FOLDER}/yandexgpt",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": max_tokens,
        },
        "messages": [
            {"role": "system", "text": "You are a helpful summarization assistant."},
            {"role": "user", "text": prompt},
        ],
    }

    resp = requests.post(YAGPT_URL, headers=HEADERS, data=json.dumps(body))
    resp.raise_for_status()
    data = resp.json()

    try:
        text = data["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:
        raise RuntimeError(f"Unexpected YandexGPT response: {data}") from e

    return text.strip()


@server.tool()
def search_docs(query: str, docs_dir: str = "docs") -> dict:
    """
    Search local docs for the query.

    Args:
        query: text to search for
        docs_dir: path to directory with .txt/.md docs

    Returns:
        {
          "matches": [
            {"path": "...", "snippet": "..."},
            ...
          ]
        }
    """
    all_docs = _load_docs_text(docs_dir)
    q = query.lower()

    matches: list[dict] = []
    for path, text in all_docs:
        if q in text.lower():
            # take first occurrence and build a small snippet
            idx = text.lower().index(q)
            start = max(0, idx - 200)
            end = min(len(text), idx + 200)
            snippet = text[start:end].replace("\n", " ")
            matches.append({"path": path, "snippet": snippet})

    # If nothing found, we still return empty list
    return {"matches": matches}


@server.tool()
def summarize_text(text: str, max_tokens: int = 300) -> dict:
    """
    Summarize text via YandexGPT (inside MCP tool).

    Args:
        text: long text to summarize
        max_tokens: approximate size of summary

    Returns:
        {"summary": "..."}
    """
    if not text.strip():
        return {"summary": ""}

    summary = _call_yandex_summarize(text, max_tokens=max_tokens)
    return {"summary": summary}


@server.tool()
def save_to_file(
    content: str,
    filename: str = "summary.txt",
    directory: str = "summaries",
    mode: str = "append",
) -> dict:
    """
    Save content to file.

    Args:
        content: text to write
        filename: name of file
        directory: folder to store file
        mode: "append" or "overwrite"

    Returns:
        {"path": "<absolute path>"}
    """
    if not content:
        raise ValueError("content is empty, nothing to save")

    os.makedirs(directory, exist_ok=True)
    full_path = os.path.abspath(os.path.join(directory, filename))

    file_mode = "a" if mode == "append" else "w"
    with open(full_path, file_mode, encoding="utf-8") as f:
        if mode == "append":
            f.write("\n\n")
        f.write(content)

    return {"path": full_path}


if __name__ == "__main__":
    # Run MCP server (stdio)
    server.run()
