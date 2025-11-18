import os
import sys
import json
import asyncio
from typing import Any, Dict, Optional

import requests

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = "yandexgpt"  # or your preferred model

# Path to your weather MCP server file
WEATHER_SERVER_PATH = "weather_mcp_server.py"


def get_env_or_die(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} environment variable")
    return value


YAC_FOLDER = get_env_or_die("YAC_FOLDER")
YAC_API_KEY = get_env_or_die("YAC_API_KEY")


def call_yandex(prompt: str, temperature: float = 0.6, max_tokens: int = 1500):
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
                "text": "You are a helpful assistant.",
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


def build_tool_aware_prompt(user_prompt: str) -> str:
    """
    System-style instructions + user question.
    We teach the model how to 'call' the weather tool via JSON.
    """
    return f"""
You are an AI assistant that can optionally use a WEATHER TOOL via an MCP client.

You have access to exactly ONE external tool:

Tool name: get_current_weather
Tool description: "Get current weather for a location using a reliable weather API."
Tool arguments (JSON object):
  - location: string (city or place name, e.g. "Amsterdam")
  - country_code: optional string (ISO country code, e.g. "NL" or "US")
  - units: string, either "metric" or "imperial"

WHEN TO USE THE TOOL
- If the user explicitly asks about *current* weather in some place (temperature, wind, etc.)
  and needs real-time data, you SHOULD call the tool.
- Otherwise (general questions, explanations, etc.) you should answer normally and NOT call the tool.

HOW TO CALL THE TOOL
- If you decide the tool is needed, you MUST respond with a SINGLE JSON object and NOTHING else.
- The JSON must have this exact shape:

{{
  "tool": "get_current_weather",
  "arguments": {{
    "location": "...",
    "country_code": null or "NL" or "US" etc,
    "units": "metric" or "imperial"
  }}
}}

Important:
- When calling the tool, output ONLY this JSON (no markdown, no text around it).
- If you don't need the tool, just answer in natural language as usual.

User question:
{user_prompt}
""".strip()


async def call_weather_tool_via_mcp(
    session: ClientSession,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call the MCP tool 'get_current_weather' and return a plain dict.

    The FastMCP server will produce structuredContent when the tool returns a dict.
    We use that when available, with text fallback.
    """
    result = await session.call_tool("get_current_weather", arguments=arguments)

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

    return {"raw": "No content returned from weather tool."}


def try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse the model's output as a JSON tool call.
    Returns dict or None.
    """
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    if data.get("tool") != "get_current_weather":
        return None

    args = data.get("arguments")
    if not isinstance(args, dict):
        return None

    return data


async def chat_loop_with_weather(session: ClientSession):
    print("=== Day 11 – YandexGPT + MCP Weather Tool ===")
    print("Ask anything. If you ask about *current weather*, the agent may call the weather tool.")
    print("Type 'exit', 'quit' or 'q' to stop.\n")

    while True:
        try:
            user_prompt = input("Your prompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_prompt.strip().lower() in {"exit", "quit", "q"}:
            print("Exiting. Bye!")
            break

        if not user_prompt.strip():
            continue  # ignore empty lines

        # ---- First: tool-planning call to YandexGPT ----
        tool_prompt = build_tool_aware_prompt(user_prompt)

        try:
            first_answer, first_tokens = call_yandex(tool_prompt)
        except Exception as e:
            print(f"\n[Error while calling YandexGPT (planning/tool decision)]\n{e}\n")
            continue

        tool_call = try_parse_tool_call(first_answer)

        if not tool_call:
            # No tool call – just treat the first answer as the final answer
            print("\nAssistant (no tool used):")
            print(first_answer)
            print("\nTokens (planning call):")
            print(
                f"inputTextTokens = {first_tokens.get('inputTextTokens')}, "
                f"completionTokens = {first_tokens.get('completionTokens')}, "
                f"totalTokens = {first_tokens.get('totalTokens')}; "
                f"(reasoningTokens = {first_tokens.get('reasoningTokens')})"
            )
            print("-" * 60)
            continue

        # ---- We have a tool call! ----
        arguments = tool_call.get("arguments", {})
        print("\n[Agent decided to call MCP tool get_current_weather with args:]")
        print(json.dumps(arguments, indent=2, ensure_ascii=False))

        try:
            weather_data = await call_weather_tool_via_mcp(session, arguments)
        except Exception as e:
            print(f"\n[Error while calling weather MCP tool]\n{e}\n")
            print("-" * 60)
            continue

        print("\n[Weather tool result (raw JSON)]:")
        print(json.dumps(weather_data, indent=2, ensure_ascii=False))

        # ---- Second: final answer call to YandexGPT with tool result ----
        final_prompt = f"""
You are a helpful assistant.

The user originally asked:
{user_prompt}

You had access to a weather tool and it returned the following JSON:

{json.dumps(weather_data, ensure_ascii=False, indent=2)}

Using ONLY this tool output (do not hallucinate other data),
answer the user's question in a clear, concise way.
If something is missing in the tool output, say that you don't know.
""".strip()

        try:
            final_answer, final_tokens = call_yandex(final_prompt)
        except Exception as e:
            print(f"\n[Error while calling YandexGPT (final answer)]\n{e}\n")
            print("-" * 60)
            continue

        print("\nAssistant (with live weather):")
        print(final_answer)

        print("\nTokens:")
        print("Planning / tool-decision call:")
        print(
            f"  inputTextTokens = {first_tokens.get('inputTextTokens')}, "
            f"completionTokens = {first_tokens.get('completionTokens')}, "
            f"totalTokens = {first_tokens.get('totalTokens')}; "
            f"(reasoningTokens = {first_tokens.get('reasoningTokens')})"
        )
        print("Final answer call (after tool):")
        print(
            f"  inputTextTokens = {final_tokens.get('inputTextTokens')}, "
            f"completionTokens = {final_tokens.get('completionTokens')}, "
            f"totalTokens = {final_tokens.get('totalTokens')}; "
            f"(reasoningTokens = {final_tokens.get('reasoningTokens')})"
        )
        print("-" * 60)


async def main_async():
    """
    Connect to the weather MCP server over stdio and run the chat loop.
    This will spawn `python weather_mcp_server.py` as a child process.
    """
    server_params = StdioServerParameters(
        command="python",
        args=[WEATHER_SERVER_PATH],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # (Optional) list tools once to sanity-check
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Connected to MCP weather server. Tools: {tool_names}\n")

            await chat_loop_with_weather(session)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except RuntimeError as e:
        # Helpful message for Windows / already running loop scenarios
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
