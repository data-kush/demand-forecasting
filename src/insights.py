"""Plain-English product briefings written by a free local LLM (Ollama), with a rule-based fallback."""
from __future__ import annotations

import json
import os
from typing import Iterator

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


def ollama_status() -> tuple[bool, str]:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
    except requests.RequestException:
        return False, "Ollama is not running"
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL for n in names):
        return False, f"model '{OLLAMA_MODEL}' not downloaded (run: ollama pull {OLLAMA_MODEL})"
    return True, f"{OLLAMA_MODEL} ready"


def build_prompt(f: dict) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in f.items())
    return (
        "You are an inventory planning analyst writing for the owner of an online gift shop.\n"
        "Using ONLY the facts below, write a short briefing for this product:\n"
        "1. One sentence on recent demand and how it compares with last year.\n"
        "2. One sentence on what the forecast expects over the next 13 weeks, including any peak.\n"
        "3. A clear stock recommendation (how many units to order now and why).\n"
        "4. One risk to watch.\n"
        "Rules: use plain business English, no jargon, at most 110 words, 4 bullet points. "
        "Quote numbers exactly as given and do not invent any new numbers.\n\n"
        f"FACTS:\n{facts}\n"
    )


def stream_briefing(facts: dict) -> Iterator[str]:
    """Yields text chunks as the local model writes them (slow on CPU, so stream it)."""
    payload = {"model": OLLAMA_MODEL, "prompt": build_prompt(facts), "stream": True,
               "options": {"temperature": 0.2, "num_predict": 260}}
    with requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break


def fallback_briefing(f: dict) -> str:
    return (
        f"- **Recent demand:** {f['Units sold, last 13 weeks']} units in the last 13 weeks "
        f"({f['Change vs same 13 weeks last year']} vs last year).\n"
        f"- **Outlook:** the model expects {f['Forecast units, next 13 weeks']} units over the next "
        f"13 weeks, peaking in the week of {f['Peak forecast week']}.\n"
        f"- **Action:** {f['Recommended order now']}.\n"
        f"- **Watch:** forecasts for this product have been off by about "
        f"{f['Typical forecast error for this product']} recently, so review weekly."
    )
