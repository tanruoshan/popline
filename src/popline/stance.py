"""RQ2: stance prompt (guideline v3 rule word for word), answer parsing, quote check. Model never sees a label."""
from __future__ import annotations

import json
import re

from rapidfuzz import fuzz

STANCES = ("PRO", "CONTRA", "NEUTRAL")

SYSTEM = ("You are a careful historian's assistant. You read German-language newspaper articles from "
          "interwar Romania (1920s) and answer only in the JSON format the user asks for.")

RULE = """Stance toward the idea of European unity or cooperation itself (not toward a person, a country or a conference):
- PRO: the text supports or welcomes it, or presents it as desirable or necessary
- CONTRA: the text rejects, mocks or warns against it
- NEUTRAL: a report without a judgement, or balanced pro and contra
The voice that counts is the text's, not that of the people it quotes: a neutral report of a pro-Europe speech is still NEUTRAL."""

USER = """Below is one newspaper article. The text comes from OCR of an old Fraktur print, so it contains recognition errors; read past them.

What is the stance of this article toward the idea of European unity or cooperation?

{rule}

Answer with one JSON object and nothing else:
{{"stance": "PRO" | "CONTRA" | "NEUTRAL", "reason": "<one English sentence>", "quote": "<the passage your answer rests on, copied exactly from the article as printed below, at most 40 words>"}}

Article:
<<<
{text}
>>>"""


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(rule=RULE, text=text)}]


def parse(content: str) -> tuple[dict | None, str | None]:
    """(answer, error). Allowed cleanup only: strip code fences and take the first JSON object.
    A wrong format is recorded as a failure and never retried or fixed by hand."""
    s = re.sub(r"^```(?:json)?|```$", "", (content or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None, "no JSON object"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg}"
    stance = str(obj.get("stance", "")).strip().upper()
    if stance not in STANCES:
        return None, f"stance not one of {STANCES}: {obj.get('stance')!r}"
    return {"stance": stance, "reason": str(obj.get("reason", "")).strip(),
            "quote": str(obj.get("quote", "")).strip()}, None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_check(quote: str, text: str, min_score: float = 90.0) -> str:
    """'exact' if the quote occurs in the article text, 'close' if a fuzzy partial match reaches
    min_score (allows small slips such as a changed hyphen), else 'not_found' (or 'empty')."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return "empty"
    if q in t:
        return "exact"
    return "close" if fuzz.partial_ratio(q, t) >= min_score else "not_found"
