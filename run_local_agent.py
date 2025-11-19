# run_local_agent.py
from __future__ import annotations

import news_weather_core as core

if __name__ == "__main__":
    core.init_db()
    try:
        core.scheduler_loop()  # блокирующий цикл, пока не Ctrl+C
    except KeyboardInterrupt:
        print("\n[NewsWeatherAgent] Stopped by user.")
