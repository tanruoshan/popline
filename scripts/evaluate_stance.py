"""RQ2 scoring. LLM vs Bea (confusion, agreement, kappa; Bea's UNSURE apart) + quote check.

In:  runs/rq2_stance_<model>.jsonl, runs/gold_articles_bea.jsonl, runs/rq2_articles.jsonl
Out: runs/rq2_results.json, runs/rq2_table.csv, evaluate_stance.run.json
Run: python scripts/evaluate_stance.py [model]
"""
from __future__ import annotations

import argparse
import collections

from popline import files
from popline.metrics import cohen_kappa
from popline.stance import STANCES, quote_check

DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"


def score(preds: dict[str, dict], gold: dict[str, dict], text: dict[str, str], model: str) -> tuple[dict, list[dict]]:
    """Per-article table and summary. Articles Bea marked UNSURE and format failures are not compared."""
    rows = []
    for art, p in preds.items():
        ans = p["answer"] or {}
        rows.append({"article": art, "bea": gold[art]["stance"], "llm": ans.get("stance"), "format_ok": p["format_ok"],
                     "quote_check": quote_check(ans.get("quote", ""), text[art]) if ans else "no answer",
                     "bea_flags": " ".join(gold[art]["flags"]), "reason": ans.get("reason", ""),
                     "quote": ans.get("quote", "")})
    sure = [r for r in rows if r["bea"] in STANCES and r["format_ok"]]
    labels = list(STANCES)
    confusion = {b: {c: sum(1 for r in sure if r["bea"] == b and r["llm"] == c) for c in labels} for b in labels}
    res = {"model": model, "articles": len(rows), "format_failures": sum(not r["format_ok"] for r in rows),
           "bea_unsure": [r["article"] for r in rows if r["bea"] == "UNSURE"],
           "compared": len(sure), "agree": sum(r["bea"] == r["llm"] for r in sure),
           "kappa": cohen_kappa([r["bea"] for r in sure], [r["llm"] for r in sure]) if sure else None,
           "confusion_bea_rows_llm_columns": confusion,
           "quote_check": dict(collections.Counter(r["quote_check"] for r in rows)),
           "host": sorted({p["host"] for p in preds.values()}),
           "host_note": sorted({p["host_note"] for p in preds.values()}),
           "model_returned": sorted({p["model_returned"] for p in preds.values()})}
    return res, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    model = ap.parse_args().model
    runs = files.runs_dir()
    stance_file = runs / f"rq2_stance_{model.replace('/', '_')}.jsonl"
    gold = {a["article"]: a for a in files.read_jsonl(runs / "gold_articles_bea.jsonl")}
    text = {a["article"]: a["text"] for a in files.read_jsonl(runs / "rq2_articles.jsonl")}
    preds = {p["article"]: p for p in files.read_jsonl(stance_file)}
    res, rows = score(preds, gold, text, model)

    outputs = [runs / "rq2_results.json", runs / "rq2_table.csv"]
    files.write_json(outputs[0], res, ensure_ascii=False)
    files.write_csv(outputs[1], rows)
    files.write_run_record("evaluate_stance", outputs, [stance_file, runs / "gold_articles_bea.jsonl",
                                                        runs / "rq2_articles.jsonl"])
    print(f"{res['articles']} articles, {res['format_failures']} format failures; Bea UNSURE: {res['bea_unsure']}")
    print(f"agreement {res['agree']} of {res['compared']}, kappa {res['kappa']}")
    print("rows = Bea, columns = LLM:", list(STANCES))
    for b in STANCES:
        print(f"  {b:8s}", [res["confusion_bea_rows_llm_columns"][b][c] for c in STANCES])
    print("quote check:", res["quote_check"])


if __name__ == "__main__":
    main()
