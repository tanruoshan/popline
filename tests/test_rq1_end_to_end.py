"""End to end: the whole RQ1 loop (all seven methods + scoring) on a synthetic mini tier.

Six passages in reading order, two IDEA articles (A: p1; B: p3 + p5, tagged implicit) and one MENTION (C: p4).
p3 holds the OCR error "Panzuropa", so only the OCR-tolerant and the dense method find B early.
"""
import math

import numpy as np
import pytest

from popline.rq1 import evaluate_tier, rankings, tier_passages

TEXTS = [
    "Das Wetter war gestern kalt und regnerisch in der ganzen Stadt.",       # p0 -
    "Die Paneuropa Bewegung fordert den Zusammenschluss der Staaten.",        # p1 A IDEA
    "Der Markt in Hermannstadt war am Samstag wieder sehr voll.",             # p2 -
    "Die Panzuropa Idee des Grafen wurde lange besprochen.",                  # p3 B IDEA (OCR error)
    "Die Weizenpreise stiegen im August deutlich an den Börsen.",             # p4 C MENTION
    "Coudenhove sprach in Wien vor sehr vielen Gästen.",                      # p5 B IDEA (continued)
]
ARTICLE = ["", "T9-01:A1", "", "T9-01:A2", "T9-01:A3", "T9-01:A2"]
GRADE = [0, 2, 0, 2, 1, 2]
Q = {"lexicon": ["paneuropa", "coudenhove"], "fuzzy_min_sim": 0.8, "b0_keyword": {"pattern": "paneurop"},
     "b0_literal": {"pattern": "paneuropa"}, "concepts": ["Die Idee eines geeinten Europas."], "rrf_k": 60}
BM25 = {"k1": 1.5, "b": 0.75, "epsilon": 0.25, "idf_scope": "tier"}
EV = {"k_values": [1, 2, 5], "effort_recall": 0.8, "relevant_grade": 2}


@pytest.fixture(scope="module")
def run():
    rows = [{"pid": f"T9-01:0:{i:03d}", "pack_id": "T9-01", "page": 0, "tier": 1, "text": t}
            for i, t in enumerate(TEXTS)]
    rows.append({"pid": "T3-01:0:000", "pack_id": "T3-01", "page": 0, "tier": 3, "text": TEXTS[1]})  # other tier
    gold_pas = {r["pid"]: {"article": ARTICLE[i] or None, "grade": GRADE[i], "in_part": True}
                for i, r in enumerate(rows[:6])}
    gold_art = {"T9-01:A1": {"tier": 1, "grade": 2, "flags": [], "stance": "PRO"},
                "T9-01:A2": {"tier": 1, "grade": 2, "flags": ["implicit"], "stance": "NEUTRAL"},
                "T9-01:A3": {"tier": 1, "grade": 1, "flags": [], "stance": None}}
    # dense: p3 is closest to the concept, then p1; the rest are unrelated
    vec = {0: [0, 1], 1: [0.8, 0.6], 2: [0, 1], 3: [1, 0], 4: [0, 1], 5: [0, 1]}
    emb = {f"T9-01:0:{i:03d}": np.array(v, dtype=float) for i, v in vec.items()}
    ps = tier_passages(rows, 1, set(gold_pas))
    methods = rankings(ps, Q, BM25, emb, np.array([[1.0, 0.0]]))
    return ps, methods, evaluate_tier(1, methods, gold_pas, gold_art, EV)


def test_only_the_tier_is_ranked(run):
    ps, methods, _ = run
    assert [p.pid for p in ps] == [f"T9-01:0:{i:03d}" for i in range(6)]
    assert [m.method for m in methods] == ["R0 reading order", "B0 keyword", "B0-literal", "B1 lexicon exact",
                                           "S1 lexicon fuzzy", "S2 dense", "S3 fusion"]


def test_first_hits_per_method(run):
    _, _, (_, first_hits, _, _) = run
    pos = {(h["method"], h["article"]): h["first_position"] for h in first_hits}
    expect = {"R0 reading order": (2, 4), "B0 keyword": (1, 4), "B0-literal": (1, 4), "B1 lexicon exact": (1, 2),
              "S1 lexicon fuzzy": (1, 2), "S2 dense": (2, 1), "S3 fusion": (1, 2)}
    for m, (a, b) in expect.items():
        assert (pos[(m, "T9-01:A1")], pos[(m, "T9-01:A2")]) == (a, b), m
    assert len(first_hits) == 7 * 2                      # MENTION article C is not a target


def test_effort_recall_and_matched(run):
    _, _, (res, _, _, _) = run
    assert (res["passages"], res["idea_articles"], res["implicit_articles"]) == (6, 2, 1)
    # 80% of 2 articles = both; effort = position of the later first hit
    assert {m: res[m]["effort_80"] for m in ("R0 reading order", "B0 keyword", "S1 lexicon fuzzy", "S2 dense")} == \
        {"R0 reading order": 4, "B0 keyword": 4, "S1 lexicon fuzzy": 2, "S2 dense": 2}
    assert res["B0 keyword"]["recall@1"] == 0.5 and res["B0 keyword"]["recall@2"] == 0.5
    assert res["S1 lexicon fuzzy"]["recall@2"] == 1.0
    assert res["B0 keyword"]["implicit_recall@2"] == 0.0 and res["S2 dense"]["implicit_recall@1"] == 1.0
    # matched passages only for lexical methods: B0 1 (p1), B1 2 (p1, p5), S1 3 (+ panzuropa)
    assert [res[m]["matched_passages"] for m in ("B0 keyword", "B1 lexicon exact", "S1 lexicon fuzzy")] == [1, 2, 3]
    assert res["R0 reading order"]["matched_passages"] is None and res["S3 fusion"]["matched_passages"] is None


def test_graded_ndcg(run):
    _, methods, (res, _, curves, tops) = run
    s2 = next(m for m in methods if m.method == "S2 dense")
    assert s2.order[:2] == ["T9-01:0:003", "T9-01:0:001"]
    dcg = lambda g: sum(x / math.log2(i + 2) for i, x in enumerate(g))  # noqa: E731
    gains = [GRADE[int(p[-3:])] for p in s2.order]
    assert res["S2 dense"]["passage_ndcg@10"] == pytest.approx(dcg(gains) / dcg(sorted(GRADE, reverse=True)))
    assert len(curves) == 7 * 6 and len(tops) == 7 * 6
