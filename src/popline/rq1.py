"""RQ1 in one place: build the method rankings of a tier and score them against the gold standard.

Used by scripts/evaluate.py (all numbers of Tables 2-4), scripts/figures.py (Fig. 4) and the end-to-end test.
Methods rank passages; an article counts as found as soon as any one of its passages is read.
"""
from __future__ import annotations

import numpy as np

from popline.metrics import article_effort, article_recall_at_k, articles_found, ndcg_at_k
from popline.pdftext import Block, Passage
from popline.retrieve import Ranking, bm25_expanded, dense, keyword_b0, reading_order_baseline, rrf


def tier_passages(rows: list[dict], tier: int, gold_pids: set[str]) -> list[Passage]:
    """Passages of one tier that the annotator read, in reading order (the order of passages.jsonl)."""
    return [Passage(r["pid"], f"{r['pack_id']}:{r['page']}", [Block("", 0, (0, 0, 1, 1), r["text"])])
            for r in rows if r["tier"] == tier and r["pid"] in gold_pids]


def rankings(ps: list[Passage], q: dict, bm25: dict, emb: dict[str, np.ndarray],
             concept_emb: np.ndarray) -> list[Ranking]:
    """All RQ1 methods on one tier. q = configs/queries.yaml, bm25 = config.yaml section bm25."""
    assert bm25.get("idf_scope", "tier") == "tier", "only IDF per ranked tier is implemented"
    params = {k: bm25[k] for k in ("k1", "b", "epsilon")}
    s1 = bm25_expanded(ps, q["lexicon"], "S1 lexicon fuzzy", q["fuzzy_min_sim"], **params)
    s2 = dense(ps, emb, concept_emb, q["concepts"])
    return [
        reading_order_baseline(ps),
        keyword_b0(ps, q["b0_keyword"]["pattern"], "B0 keyword"),
        keyword_b0(ps, q["b0_literal"]["pattern"], "B0-literal"),      # added after first results (q1b)
        bm25_expanded(ps, q["lexicon"], "B1 lexicon exact", **params),
        s1, s2, rrf([s1, s2], q["rrf_k"], "S3 fusion", [p.pid for p in ps]),
    ]


def evaluate_tier(tier: int, methods: list[Ranking], gold_pas: dict[str, dict], gold_art: dict[str, dict],
                  ev: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Scores for one tier: (results, first hits, found-curves, top 10 with "why"). ev = config.yaml evaluation."""
    order = methods[0].order
    art_of = {p: gold_pas[p]["article"] for p in order}
    idea = {a for a, v in gold_art.items() if v["tier"] == tier and v["grade"] == ev["relevant_grade"]}
    implicit = {a for a in idea if "implicit" in gold_art[a]["flags"]}
    # graded gain: IDEA 2, MENTION 1; a passage outside the marked part of an IDEA article counts as MENTION at most
    gain = {p: (gold_pas[p]["grade"] if gold_pas[p]["in_part"] else min(gold_pas[p]["grade"], 1)) for p in order}
    results: dict = {"passages": len(order), "idea_articles": len(idea), "implicit_articles": len(implicit)}
    first_hits, curves, tops = [], [], []
    for m in methods:
        curve = articles_found(m.order, art_of, idea)
        pos: dict[str, int] = {}
        for i, p in enumerate(m.order, 1):
            a = art_of[p]
            if a in idea and a not in pos:
                pos[a] = i
        res = {"effort_80": article_effort(curve, len(idea), ev["effort_recall"]),
               "effort_100": article_effort(curve, len(idea), 1.0),
               **{f"recall@{k}": article_recall_at_k(curve, len(idea), k) for k in ev["k_values"]},
               "passage_ndcg@10": ndcg_at_k([gain[p] for p in m.order], 10),
               "matched_passages": sum(1 for p in m.order if m.scores[p] > 0) if m.kind == "lexical" else None}
        if implicit:
            ic = articles_found(m.order, art_of, implicit)
            res.update({f"implicit_recall@{k}": article_recall_at_k(ic, len(implicit), k) for k in ev["k_values"]})
        results[m.method] = res
        for a in sorted(idea):
            first_hits.append({"tier": tier, "article": a, "method": m.method, "first_position": pos.get(a),
                               "implicit": a in implicit, "stance": gold_art[a]["stance"]})
        curves += [{"tier": tier, "method": m.method, "n_read": i, "found": c} for i, c in enumerate(curve, 1)]
        tops += [{"tier": tier, "method": m.method, "rank": i, "pid": p, "article": art_of[p],
                  "grade": gold_pas[p]["grade"], "why": m.why.get(p, "")} for i, p in enumerate(m.order[:10], 1)]
    return results, first_hits, curves, tops
