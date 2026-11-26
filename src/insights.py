"""Plain-English writing by a free local LLM (Ollama): product briefings, an executive summary,
supplier emails and a Q&A assistant. Every prompt is grounded in numbers computed by the app."""
from __future__ import annotations

import json
import os
from typing import Iterator

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
GROUNDING = ("Use ONLY the facts provided. Quote numbers exactly as given and never invent numbers, "
             "products, dates or names. If the facts do not answer something, say so plainly.")


def ollama_status() -> tuple[bool, str]:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
    except requests.RequestException:
        return False, "Ollama is not running"
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL for n in names):
        return False, f"model '{OLLAMA_MODEL}' not downloaded (run: ollama pull {OLLAMA_MODEL})"
    return True, f"{OLLAMA_MODEL} ready"


def stream_chat(messages: list[dict], max_words: int = 200, temperature: float = 0.2) -> Iterator[str]:
    """Stream a reply from the local model (CPU is slow, so show words as they arrive)."""
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True,
               "options": {"temperature": temperature, "num_predict": int(max_words * 1.8),
                           "num_ctx": 4096}}
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    break


def _facts(f: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in f.items())


# ---------- product briefing ----------
def stream_briefing(facts: dict) -> Iterator[str]:
    return stream_chat([
        {"role": "system", "content": "You are an inventory planning analyst writing for the owner of an "
                                      "online gift shop. " + GROUNDING},
        {"role": "user", "content":
            "Write a short briefing for this product as 4 bullet points: (1) recent demand vs last year, "
            "(2) what the forecast expects over the next 13 weeks including the peak, (3) a clear stock "
            "recommendation, (4) one risk to watch. Plain business English, at most 110 words, "
            "no introduction line.\n\nFACTS:\n" + _facts(facts)},
    ], max_words=130)


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


# ---------- executive summary ----------
def stream_exec_summary(context: str) -> Iterator[str]:
    return stream_chat([
        {"role": "system", "content": "You are a senior retail analyst briefing a busy shop owner. " + GROUNDING},
        {"role": "user", "content":
            "Write an executive summary with exactly these three bold headings:\n"
            "**What is happening** (2 sentences on the outlook and trends)\n"
            "**What to do this week** (3 short bullet points naming specific products)\n"
            "**Why trust the forecast** (1 sentence on the test results)\n"
            "At most 170 words, no introduction line.\n\nFACTS:\n" + context},
    ], max_words=190)


# ---------- supplier email ----------
def stream_po_email(lines: list[str], lead_time: int, total_units: float, total_value: float) -> Iterator[str]:
    order = "\n".join(lines)
    return stream_chat([
        {"role": "system", "content": "You write clear, polite purchasing emails for a small UK online "
                                      "gift retailer. " + GROUNDING},
        {"role": "user", "content":
            f"Draft an email to our supplier placing this week's order. Ask them to confirm delivery within "
            f"{lead_time} week(s). Include the order lines exactly as given, as a simple list. Total: "
            f"{total_units:,.0f} units, about £{total_value:,.0f} at our usual prices. Start with "
            "'Subject:' then the email body. Sign off as 'Purchasing team'. Keep the text around the list "
            f"under 80 words.\n\nORDER LINES:\n{order}"},
    ], max_words=60 + 12 * len(lines))


# ---------- Q&A ----------
def qa_messages(context: str, product_facts: list[dict], history: list[dict], question: str) -> list[dict]:
    facts = "BUSINESS FACTS:\n" + context
    for f in product_facts:
        facts += "\n\nPRODUCT FACTS:\n" + _facts(f)
    system = ("You are the analyst assistant inside a demand-forecasting dashboard for a UK online gift "
              "retailer. Answer the owner's question in at most 120 words, in plain English, using short "
              "bullet points when listing things. Always explain WHY, citing 2-3 relevant facts (for example "
              "recent sales, the forecast, safety stock or buyer concentration). " + GROUNDING + " If the question is about a product not "
              "in the facts, suggest they name the product or open the Product planner tab.\n\n" + facts)
    return [{"role": "system", "content": system}, *history[-4:], {"role": "user", "content": question}]
