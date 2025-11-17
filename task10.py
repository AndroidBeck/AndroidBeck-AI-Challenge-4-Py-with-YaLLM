import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # --- 1. Describe how to run the time MCP server ---
    # Option A: using pip-installed module
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server_time"],   # this runs the Time MCP Server
        env=None,                         # add env vars here if needed
    )

    # Option B (if you prefer uvx):
    # server_params = StdioServerParameters(
    #     command="uvx",
    #     args=["mcp-server-time"],
    # )

    async with AsyncExitStack() as stack:
        # --- 2. Open stdio transport (subprocess) ---
        read, write = await stack.enter_async_context(stdio_client(server_params))

        # --- 3. Create a ClientSession and initialize MCP ---
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # --- 4. List tools exposed by the Time MCP Server ---
        tools_response = await session.list_tools()
        tools = tools_response.tools

        print("=== Tools exposed by Time MCP server ===")
        if not tools:
            print("No tools found.")
        else:
            for i, tool in enumerate(tools, start=1):
                print(f"{i}. {tool.name}")
                if getattr(tool, "description", None):
                    print(f"   description: {tool.description}")
                if getattr(tool, "inputSchema", None):
                    print(f"   input schema: {tool.inputSchema}")
                print()

        # --- 5. Call get_current_time for Europe/Amsterdam ---
        print("=== Calling get_current_time(Europe/Amsterdam) ===")
        result = await session.call_tool(
            "get_current_time",
            arguments={"timezone": "Europe/Amsterdam"},
        )

        # result.content is a list of content blocks (usually TextContent or structured)
        # For this server, structured output is typically in result.structuredContent or JSON in text.
        print("Raw CallToolResult:")
        print(result)

        # If it returns structured data, you might do something like:
        # print("Structured:", result.structuredContent)
        # If it's just text, something like:
        # if result.content:
        #     print("Text:", getattr(result.content[0], "text", None))


if __name__ == "__main__":
    asyncio.run(main())
