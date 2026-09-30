"""OCR quality + RQ1 by OCR quality.

Passage quality = share of words (3+ chars, not numbers) found in wordfreq German (zipf > 0); letter-digit
mixes count as errors. Article = word-weighted mean. Tier-1 IDEA articles split at median -> low / high.
In:  runs/passages.jsonl, runs/gold_*_bea.jsonl, runs/rq1_first_hits.csv
Out: runs/ocr_quality.csv, runs/rq1_ocr.json, ocr_quality.run.json
Run: python scripts/ocr_quality.py
"""
from __future__ import annotations

import re
import statistics

from wordfreq import zipf_frequency

from popline import files


def quality(text):
    toks = [t for t in re.findall(r"\w+", text.lower()) if len(t) >= 3 and not t.isdigit()]
    ok = sum(1 for t in toks if t.isalpha() and zipf_frequency(t, "de") > 0)
    return ok / len(toks) if toks else None, len(toks)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):                                  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def main() -> None:
    runs = files.runs_dir()
    rows = files.read_jsonl(runs / "passages.jsonl")
    q = {}
    table = []
    for r in rows:
        qual, n = quality(r["text"])
        q[r["pid"]] = (qual, n)
        table.append({"pid": r["pid"], "tier": r["tier"], "tokens": n,
                      "quality": "" if qual is None else round(qual, 4)})
    files.write_csv(runs / "ocr_quality.csv", table)

    gold_p = {p["pid"]: p for p in files.read_jsonl(runs / "gold_passages_bea.jsonl")}
    arts = {a["article"]: a for a in files.read_jsonl(runs / "gold_articles_bea.jsonl")}
    art_q = {}
    for a in arts:
        parts = [q[p] for p, v in gold_p.items() if v["article"] == a and q[p][0] is not None]
        art_q[a] = sum(x * n for x, n in parts) / sum(n for _, n in parts) if parts else None

    res = {"passages": {}, "tier1_idea": {}}
    for tier in (1, 3):
        vals = [q[r["pid"]][0] for r in rows if r["tier"] == tier and q[r["pid"]][0] is not None]
        res["passages"][f"tier{tier}"] = {"n": len(vals), "median": round(statistics.median(vals), 3),
                                          "q1": round(statistics.quantiles(vals, n=4)[0], 3),
                                          "q3": round(statistics.quantiles(vals, n=4)[2], 3)}

    idea = sorted(a for a, v in arts.items() if v["tier"] == 1 and v["grade"] == 2)
    cut = statistics.median(art_q[a] for a in idea)
    group = {a: ("low" if art_q[a] < cut else "high") for a in idea}
    hits = [h for h in files.read_csv(runs / "rq1_first_hits.csv") if h["tier"] == "1"]
    res["tier1_idea"]["median_quality"] = round(cut, 3)
    res["tier1_idea"]["articles"] = {a: {"quality": round(art_q[a], 3), "group": group[a]} for a in idea}
    for m in sorted({h["method"] for h in hits}):
        pos = {h["article"]: int(h["first_position"]) for h in hits if h["method"] == m}
        out = {}
        for g in ("low", "high"):
            ps = [pos[a] for a in idea if group[a] == g]
            out[g] = {"n": len(ps), "median_first_hit": statistics.median(ps),
                      "found_in_top20": sum(p <= 20 for p in ps)}
        out["spearman_quality_vs_position"] = round(spearman([art_q[a] for a in idea], [pos[a] for a in idea]), 3)
        res["tier1_idea"][m] = out
    files.write_json(runs / "rq1_ocr.json", res)
    files.write_run_record("ocr_quality", [runs / "ocr_quality.csv", runs / "rq1_ocr.json"],
                           [runs / "passages.jsonl", runs / "gold_passages_bea.jsonl", runs / "rq1_first_hits.csv"])

    print("passage OCR quality:", res["passages"])
    print(f"tier 1 IDEA articles, median quality {cut:.3f} (low < median <= high)")
    print(f"{'method':20s} {'low: median pos':>16s} {'top20':>6s} {'high: median pos':>17s} {'top20':>6s} {'rho':>7s}")
    for m in sorted({h["method"] for h in hits}):
        v = res["tier1_idea"][m]
        lo, hi = v["low"], v["high"]
        print(f"{m:20s} {lo['median_first_hit']:>16} {lo['found_in_top20']:>3}/{lo['n']:<2} "
              f"{hi['median_first_hit']:>17} {hi['found_in_top20']:>3}/{hi['n']:<2} "
              f"{v['spearman_quality_vs_position']:>7}")


if __name__ == "__main__":
    main()
