"""RQ1 methods. Each returns ranked passage ids + a "why" per passage (shown to the historian).

Ties keep reading order (same rule for every method).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz.distance import Levenshtein

from popline.pdftext import Passage, tokens

KINDS = ("baseline", "lexical", "dense", "fusion")


@dataclass
class Ranking:
    method: str
    order: list[str]                 # passage ids, best first
    scores: dict[str, float]
    why: dict[str, str]              # passage id -> short explanation
    kind: str = "lexical"            # one of KINDS; only lexical methods have a "matched" set (score > 0)

    def __post_init__(self) -> None:
        assert self.kind in KINDS, f"unknown kind {self.kind!r}"


def _rank(method: str, pids: list[str], scores: dict[str, float], why: dict[str, str], kind: str) -> Ranking:
    order = sorted(pids, key=lambda p: -scores[p])     # sorted() is stable: ties keep reading order
    return Ranking(method, order, scores, why, kind)


def reading_order_baseline(passages: list[Passage], method: str = "R0 reading order") -> Ranking:
    """R0: no ranking, the passages as a reader meets them on the page."""
    pids = [p.pid for p in passages]
    return Ranking(method, pids, {p: 0.0 for p in pids}, {p: "" for p in pids}, "baseline")


def query_terms(lexicon: list[str]) -> list[str]:
    return sorted({t for entry in lexicon for t in tokens(entry)})


def expand_exact(terms: list[str], vocab: set[str], min_prefix: int = 5) -> dict[str, str]:
    """Corpus token -> query term. Terms of 5+ letters also match as prefix (German endings). Exact spelling."""
    out = {}
    for tok in vocab:
        for t in terms:
            if tok == t or (len(t) >= min_prefix and tok.startswith(t)):
                out[tok] = t
    return out


def expand_fuzzy(terms: list[str], vocab: set[str], min_sim: float, min_prefix: int = 5) -> dict[str, str]:
    """As expand_exact, plus OCR variants: token prefix vs term, normalised Levenshtein >= min_sim."""
    out = expand_exact(terms, vocab, min_prefix)
    for tok in vocab - set(out):
        for t in terms:
            if len(t) >= min_prefix and len(tok) >= len(t) - 1 and \
                    Levenshtein.normalized_similarity(tok[: len(t)], t) >= min_sim:
                out[tok] = t
                break
    return out


def keyword_b0(passages: list[Passage], prefix: str, method: str = "B0 keyword") -> Ranking:
    """B0: archive-style search. Score = number of tokens starting with the keyword."""
    scores, why = {}, {}
    for p in passages:
        hits = [t for t in tokens(p.text) if t.startswith(prefix)]
        scores[p.pid], why[p.pid] = float(len(hits)), ", ".join(sorted(set(hits)))
    return _rank(method, [p.pid for p in passages], scores, why, "lexical")


def bm25_expanded(passages: list[Passage], lexicon: list[str], method: str, fuzzy_sim: float | None = None,
                  k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> Ranking:
    """B1 (exact lexicon) and S1 (OCR-tolerant lexicon): BM25 over the query expanded to corpus tokens.

    IDF comes from `passages` only, i.e. from the set that is ranked (one tier). k1, b, epsilon: see
    configs/config.yaml (bm25); the defaults are those of rank_bm25 0.2.2."""
    docs = [tokens(p.text) for p in passages]
    vocab = {t for d in docs for t in d}
    terms = query_terms(lexicon)
    mapping = expand_fuzzy(terms, vocab, fuzzy_sim) if fuzzy_sim else expand_exact(terms, vocab)
    bm25 = BM25Okapi(docs, k1=k1, b=b, epsilon=epsilon)
    raw = bm25.get_scores(list(mapping))
    scores, why = {}, {}
    for p, d, s in zip(passages, docs, raw):
        hits = sorted({t for t in d if t in mapping})
        scores[p.pid] = float(s) if hits else 0.0
        why[p.pid] = ", ".join(f"{t}" if mapping[t] == t else f"{t}~{mapping[t]}" for t in hits)
    return _rank(method, [p.pid for p in passages], scores, why, "lexical")


def dense(passages: list[Passage], emb: dict[str, np.ndarray], concept_emb: np.ndarray,
          concepts: list[str], method: str = "S2 dense") -> Ranking:
    """S2: cosine similarity to the closest concept description (vectors are L2-normalised)."""
    scores, why = {}, {}
    for p in passages:
        sims = concept_emb @ emb[p.pid]
        j = int(np.argmax(sims))
        scores[p.pid], why[p.pid] = float(sims[j]), f"close to concept {j + 1}: {concepts[j][:50]}..."
    return _rank(method, [p.pid for p in passages], scores, why, "dense")


def rrf(rankings: list[Ranking], k: int, method: str, reading_order: list[str]) -> Ranking:
    """Reciprocal rank fusion (Cormack et al. 2009). Credit only for passages a method retrieved (score > 0)."""
    ranks = [{p: i for i, p in enumerate(r.order)} for r in rankings]
    scores = {p: sum(1.0 / (k + rk[p] + 1) for r, rk in zip(rankings, ranks) if r.scores[p] > 0)
              for p in reading_order}
    why = {p: " | ".join(r.why[p] for r in rankings if r.why.get(p)) for p in reading_order}
    return _rank(method, reading_order, scores, why, "fusion")
