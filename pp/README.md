The DB layer is isolated in db.py – schema + CRUD + stats.

The llm call is isolated in llm_client.py.

The higher-level behavior (normal chat vs summarization) is in chat_logic.py.

The while True loop in main.py is now short and easy to read.

Token print logic is shared in _print_usage.