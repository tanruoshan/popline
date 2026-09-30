"""RQ1. Each method ranks all passages of a tier; article found when any of its passages is read.

Methods: R0 reading order, B0 "paneurop*", B0-literal "paneuropa*" (added after first results),
B1 lexicon exact, S1 lexicon fuzzy, S2 dense, S3 fusion(S1, S2). Settings: configs/queries.yaml, configs/config.yaml.
The ranking and scoring code is in src/popline/rq1.py; this script only reads and writes files.
In:  runs/passages.jsonl, runs/gold_*_bea.jsonl, runs/emb_multilingual-e5-base.npz
Out: runs/rq1_results.json, rq1_first_hits.csv, rq1_curves.csv, rq1_top10.jsonl, evaluate.run.json
Run: python scripts/evaluate.py
"""
from __future__ import annotations

import numpy as np

from popline import files
from popline.rq1 import evaluate_tier, rankings, tier_passages


def print_table(results: dict) -> None:
    for tier, res in results.items():
        print(f"\n{tier}: {res['passages']} passages, {res['idea_articles']} IDEA articles "
              f"({res['implicit_articles']} implicit)")
        cols = ["effort_80", "effort_100", "recall@5", "recall@10", "recall@20", "passage_ndcg@10"]
        if res["implicit_articles"]:
            cols += ["implicit_recall@20"]
        print(f"{'method':20s}" + "".join(f"{c:>14s}" for c in cols))
        for m, v in res.items():
            if isinstance(v, dict):
                print(f"{m:20s}" + "".join(f"{'-' if v.get(c) is None else round(v[c], 3):>14}" for c in cols))


def main() -> None:
    cfg, q = files.config(), files.queries()
    runs = files.runs_dir(cfg)
    ev = cfg["evaluation"]
    rows = files.read_jsonl(runs / "passages.jsonl")
    gold_art = {a["article"]: a for a in files.read_jsonl(runs / "gold_articles_bea.jsonl")}
    gold_pas = {p["pid"]: p for p in files.read_jsonl(runs / "gold_passages_bea.jsonl")}
    emb_path = runs / f"emb_{q['dense_model'].split('/')[-1]}.npz"
    with np.load(emb_path) as e:
        emb = dict(zip(e["pids"].tolist(), e["passages"]))
        concepts = e["concepts"]
    assert len(emb) == len(rows) and all(r["pid"] in emb for r in rows), "embeddings do not match passages.jsonl"

    results, first_hits, curves, tops = {}, [], [], []
    for tier in sorted({a["tier"] for a in gold_art.values()} | {3}):
        ps = tier_passages(rows, tier, set(gold_pas))
        if not ps:
            continue
        res, fh, cu, tp = evaluate_tier(tier, rankings(ps, q, cfg["bm25"], emb, concepts), gold_pas, gold_art, ev)
        results[f"tier{tier}"] = res
        first_hits += fh
        curves += cu
        tops += tp

    outputs = [runs / n for n in ("rq1_results.json", "rq1_first_hits.csv", "rq1_curves.csv", "rq1_top10.jsonl")]
    files.write_json(outputs[0], results)
    files.write_csv(outputs[1], first_hits)
    files.write_csv(outputs[2], curves)
    files.write_jsonl(outputs[3], tops)
    files.write_run_record("evaluate", outputs, [runs / "passages.jsonl", runs / "gold_articles_bea.jsonl",
                                                 runs / "gold_passages_bea.jsonl", emb_path])
    print_table(results)
    print("\nwritten: runs/rq1_results.json, rq1_first_hits.csv, rq1_curves.csv, rq1_top10.jsonl, evaluate.run.json")


if __name__ == "__main__":
    main()
