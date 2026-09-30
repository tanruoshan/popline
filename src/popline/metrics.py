"""Ranking + agreement measures: article recall@k / reading effort, passage nDCG@k / recall@k, Cohen's kappa."""
from __future__ import annotations

import math
from collections import Counter


def dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], k: int) -> float | None:
    """Graded nDCG (Järvelin and Kekäläinen 2002); gain = grade (MENTION 1, IDEA 2).
    None if the list holds no relevant item at all."""
    ideal = dcg(sorted(ranked_grades, reverse=True), k)
    return dcg(ranked_grades, k) / ideal if ideal > 0 else None


def recall_at_k(ranked_rel: list[bool], k: int) -> float | None:
    total = sum(ranked_rel)
    return sum(ranked_rel[:k]) / total if total else None


def reading_effort(ranked_rel: list[bool], target: float) -> int | None:
    """How many passages a reader goes through, top down, to find `target` share of the relevant ones."""
    total = sum(ranked_rel)
    if not total:
        return None
    need, found = math.ceil(target * total), 0
    for i, r in enumerate(ranked_rel, 1):
        found += r
        if found >= need:
            return i
    return len(ranked_rel)


def cohen_kappa(a: list, b: list) -> float | None:
    assert len(a) == len(b), "both annotators must label the same items"
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def articles_found(order: list[str], passage_article: dict[str, str | None], relevant: set[str]) -> list[int]:
    """For each reading position n (1-based), how many relevant articles were hit by the first n
    passages. An article counts as found as soon as any one of its passages is read."""
    seen: set[str] = set()
    curve = []
    for pid in order:
        a = passage_article.get(pid)
        if a in relevant:
            seen.add(a)
        curve.append(len(seen))
    return curve


def article_recall_at_k(curve: list[int], n_relevant: int, k: int) -> float | None:
    if not n_relevant:
        return None
    return curve[min(k, len(curve)) - 1] / n_relevant if curve else 0.0


def article_effort(curve: list[int], n_relevant: int, target: float) -> int | None:
    """Passages a historian reads, top down, until `target` share of the relevant articles is found."""
    if not n_relevant:
        return None
    need = math.ceil(target * n_relevant)
    for i, found in enumerate(curve, 1):
        if found >= need:
            return i
    return None      # never reached: some relevant articles have no passage in the ranking
