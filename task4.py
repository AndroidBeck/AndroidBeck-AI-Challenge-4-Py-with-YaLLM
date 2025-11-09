#!/usr/bin/env python3
"""
Day 3 – Multi‑pass LLM Orchestrator

Features (single interactive run):
1) Straight answer
2) Step‑by‑step answer
3) Prompt‑engineering round-trip
   3.1) Ask LLM to craft the best prompt
   3.2) Call LLM with that prompt
4) Panel of experts – you enter a comma‑separated list and each "expert" answers

Supports two providers via env var PROVIDER={YANDEX|OPENAI}
- Yandex:  YAC_API_KEY, YAC_FOLDER, YAC_MODEL (default: yandexgpt)
- OpenAI:  OPENAI_API_KEY, OPENAI_MODEL (default: gpt-4o-mini)

Usage:
  $ export PROVIDER=YANDEX  # or OPENAI
  $ export YAC_API_KEY=...; export YAC_FOLDER=...
  $ python day3_orchestrator.py

Notes:
- Keeps messages minimal per step for clear separation of roles.
- After each step, you can choose to continue (y/n).
"""
import os
import sys
import json
import textwrap
from typing import List, Dict, Optional
import requests

# -----------------------------
# Core Client Abstraction
# -----------------------------
class LLMClient:
    def send(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

# -----------------------------
# Yandex GPT (REST) client
# -----------------------------
class YandexClient(LLMClient):
    def __init__(self):
        self.url = os.getenv("YAGPT_URL", "https://llm.api.cloud.yandex.net/foundationModels/v1/completion")
        self.api_key = os.getenv("YAC_API_KEY")
        self.folder = os.getenv("YAC_FOLDER")
        self.model = os.getenv("YAC_MODEL", "yandexgpt")
        if not self.api_key or not self.folder:
            raise RuntimeError("Missing YAC_API_KEY or YAC_FOLDER env vars for Yandex provider.")

    def send(self, messages: List[Dict[str, str]]) -> str:
        # Yandex expects {role, text} items and modelUri like gpt://<folder>/<model>
        y_messages = []
        for m in messages:
            # Map OpenAI-style {role, content} to Yandex {role, text}
            if "content" in m:
                y_messages.append({"role": m["role"], "text": m["content"]})
            else:
                y_messages.append(m)

        payload = {
            "modelUri": f"gpt://{self.folder}/{self.model}",
            "completionOptions": {
                "temperature": 0.6,
                "maxTokens": 1200,
                "stream": False,
            },
            "messages": y_messages,
        }
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Typical path: result.alternatives[0].message.text
        try:
            return data["result"]["alternatives"][0]["message"]["text"].strip()
        except Exception:
            return json.dumps(data, ensure_ascii=False)

# -----------------------------
# OpenAI (REST) client
# -----------------------------
class OpenAIClient(LLMClient):
    def __init__(self):
        self.url = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY env var for OpenAI provider.")

    def send(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 1200,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return json.dumps(data, ensure_ascii=False)

# -----------------------------
# Helpers
# -----------------------------

def build_client() -> LLMClient:
    provider = os.getenv("PROVIDER", "YANDEX").upper()
    if provider == "YANDEX":
        return YandexClient()
    elif provider == "OPENAI":
        return OpenAIClient()
    else:
        raise RuntimeError(f"Unsupported PROVIDER: {provider}")


def prompt_continue(step_label: str) -> bool:
    while True:
        ans = input(f"\nContinue to {step_label}? (y/n): ").strip().lower()
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"): return False
        print("Please type 'y' or 'n'.")


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# -----------------------------
# Step builders
# -----------------------------

def step1_straight_answer(client: LLMClient, task: str) -> str:
    messages = [
        {"role": "system", "content": "Answer the user's task concisely and directly. Avoid preambles."},
        {"role": "user", "content": task},
    ]
    return client.send(messages)


def step2_step_by_step(client: LLMClient, task: str) -> str:
    messages = [
        {"role": "system", "content": (
            "Provide a clear, step-by-step solution. Number each step. "
            "State any critical assumptions. Keep it practical and executable."
        )},
        {"role": "user", "content": task},
    ]
    return client.send(messages)


def step3_1_make_prompt(client: LLMClient, task: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a prompt engineer. Craft the single best prompt to solve the user's task. "
            "Keep it self-contained (include the task), specify role, goals, constraints, desired format, and evaluation criteria. "
            "Return ONLY the prompt text, no surrounding commentary."
        )},
        {"role": "user", "content": task},
    ]
    return client.send(messages)


def step3_2_use_prompt(client: LLMClient, engineered_prompt: str, task: str) -> str:
    messages = [
        {"role": "system", "content": "Follow the user's prompt precisely. Produce the requested deliverable."},
        {"role": "user", "content": engineered_prompt + "\n\n(Note: Original task context preserved.)\n" + task},
    ]
    return client.send(messages)


def step4_experts_panel(client: LLMClient, task: str, experts_csv: str) -> str:
    experts = [e.strip() for e in experts_csv.split(",") if e.strip()]
    if not experts:
        return "No experts entered."

    outputs = []
    for expert in experts:
        messages = [
            {"role": "system", "content": (
                f"You are '{expert}', a domain expert on this task. "
                "Provide a focused answer: 3-7 bullet points with justification, then a 1-paragraph recommendation."
            )},
            {"role": "user", "content": task},
        ]
        ans = client.send(messages)
        outputs.append(f"---\nExpert: {expert}\n\n{ans}\n")

    # (Optional) Synthesize final panel consensus
    synthesis_prompt = "\n\n".join([f"Expert {i+1} ({experts[i]}):\n{outputs[i]}" for i in range(len(experts))])
    synth_messages = [
        {"role": "system", "content": (
            "Act as a neutral facilitator. Given the experts' responses, produce a brief synthesis: "
            "key agreements, disagreements, risks, and a final recommended plan with next steps."
        )},
        {"role": "user", "content": synthesis_prompt},
    ]
    synthesis = client.send(synth_messages)

    panel = "\n".join(outputs) + "\n===\nPanel synthesis\n\n" + synthesis
    return panel

# -----------------------------
# Main
# -----------------------------

def main():
    print_section("Day 4 – Multi‑pass LLM Orchestrator")
    client = build_client()

    task = input("Enter the Task: ").strip()
    if not task:
        print("No task provided. Exiting.")
        sys.exit(0)

    # 1) Straight answer
    print_section("1. Giving the straight answer")
    ans1 = step1_straight_answer(client, task)
    print(ans1)
    if not prompt_continue("step 2 (step-by-step answer)"):
        print("Stopped.")
        return

    # 2) Step-by-step
    print_section("2. Giving a step-by-step answer")
    ans2 = step2_step_by_step(client, task)
    print(ans2)
    if not prompt_continue("step 3.1 (ask LLM to create a prompt)"):
        print("Stopped.")
        return

    # 3.1) Prompt engineering
    print_section("3.1 Asking LLM to create a prompt")
    engineered_prompt = step3_1_make_prompt(client, task)
    print(engineered_prompt)
    if not prompt_continue("step 3.2 (use this prompt)"):
        print("Stopped.")
        return

    # 3.2) Use engineered prompt
    print_section("3.2 Using this prompt")
    ans3 = step3_2_use_prompt(client, engineered_prompt, task)
    print(ans3)
    if not prompt_continue("step 4 (experts panel)"):
        print("Stopped.")
        return

    # 4) Experts
    print_section("4: Enter the experts (comma-separated, e.g., 'PM, Architect, SRE, Data Scientist')")
    experts_line = input("Experts: ")

    print_section("4. Experts' answers")
    panel = step4_experts_panel(client, task, experts_line)
    print(panel)

    print("\nAll steps completed. Goodbye!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
