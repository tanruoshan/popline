"""RQ2. LLM stance per IDEA article: PRO/CONTRA/NEUTRAL + reason + quote. Guideline rule in the prompt.

Endpoint in .env at the repo root (see .env.example; key never printed). Temperature 0, max_tokens 400.
Network errors retried; wrong format stored as failure, never retried. Rerun skips finished articles.
In:  runs/rq2_articles.jsonl
Out: runs/rq2_stance_<model>.jsonl (one run record per line: model, host, tokens, raw answer)
Run: python scripts/stance.py --dry-run   (show prompt)   |   python scripts/stance.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from urllib.parse import urlparse

from popline import files
from popline.stance import messages, parse

TEMPERATURE, MAX_TOKENS, RETRIES = 0.0, 400, 5
DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and stop")
    ap.add_argument("--limit", type=int, default=None, help="only the first N articles (a test run)")
    args = ap.parse_args()

    runs = files.runs_dir()
    articles = files.read_jsonl(runs / "rq2_articles.jsonl")
    if args.limit:
        articles = articles[: args.limit]
    if args.dry_run:
        for m in messages(articles[0]["text"]):
            print(f"--- {m['role']} ---\n{m['content'][:3000]}")
        print(f"\n({len(articles)} articles would be sent)")
        return

    # imported here so that --dry-run and the tests work without the client libraries
    from dotenv import load_dotenv
    from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

    load_dotenv(files.ROOT / ".env")
    base_url = os.environ["POPLINE_LLM_BASE_URL"]
    model = os.environ.get("POPLINE_LLM_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url=base_url, api_key=os.environ["POPLINE_LLM_API_KEY"], timeout=180)
    out = runs / f"rq2_stance_{model.replace('/', '_')}.jsonl"
    done = {r["article"] for r in files.read_jsonl(out)} if out.exists() else set()
    print(f"{len(articles)} articles, {len(done)} already done; model {model} at {urlparse(base_url).netloc}")

    for a in articles:
        if a["article"] in done:
            continue
        t0 = time.time()
        for attempt in range(RETRIES):
            try:
                r = client.chat.completions.create(model=model, messages=messages(a["text"]),
                                                   temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                break
            except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError) as e:
                wait = 2 ** (attempt + 1)
                print(f"  {a['article']}: {type(e).__name__}, retry in {wait} s")
                time.sleep(wait)
        else:
            print(f"  {a['article']}: gave up after {RETRIES} tries; rerun later")
            continue
        content = r.choices[0].message.content or ""
        answer, error = parse(content)
        if r.choices[0].finish_reason == "length":
            answer, error = None, "cut off (finish_reason length)"
        row = {"article": a["article"], "model_requested": model, "model_returned": r.model,
               "host": urlparse(base_url).netloc, "host_note": os.environ.get("POPLINE_LLM_HOST_NOTE", ""),
               "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "finish_reason": r.choices[0].finish_reason,
               "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
               "completion_tokens": getattr(r.usage, "completion_tokens", None),
               "format_ok": answer is not None, "error": error, "answer": answer, "raw": content,
               "seconds": round(time.time() - t0, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {a['article']}: {answer['stance'] if answer else 'FORMAT FAIL: ' + error} ({row['seconds']} s)")
    files.write_run_record("stance", [out], [runs / "rq2_articles.jsonl"],
                           {"model": model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
