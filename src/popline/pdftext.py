"""PDF OCR layer -> passages. Passage = text blocks stacked in one column (min/max words from config).

Glyph normalisation only (long s, ligatures, quotes, line-end hyphens). Never OCR correction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

GLYPHS = {"ſ": "s", "ﬅ": "st", "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
          "„": '"', "“": '"', "”": '"', "‚": "'", "‘": "'", "’": "'", "¬": "-"}


def normalize(text: str) -> str:
    """Glyph normalisation and de-hyphenation. Not an OCR correction."""
    for old, new in GLYPHS.items():
        text = text.replace(old, new)
    text = re.sub(r"(\w)-\s*\n\s*([a-zäöüß])", r"\1\2", text)   # Ver-\nhandlung -> Verhandlung
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class Block:
    page_key: str            # "<pack_id>:<page index>"
    idx: int
    bbox: tuple[float, float, float, float]
    text: str

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class Passage:
    pid: str                 # "<pack_id>:<page index>:<n>"
    page_key: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return normalize("\n".join(b.text for b in self.blocks))

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def page_blocks(page: pymupdf.Page, page_key: str) -> list[Block]:
    out = []
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks"):
        if kind == 0 and text.strip():
            out.append(Block(page_key, len(out), (x0, y0, x1, y1), text))
    return out


def _below(a: Block, b: Block, gap: float = 25.0) -> bool:
    """True if b starts just below a and overlaps it horizontally by half the narrower width."""
    ax0, _, ax1, ay1 = a.bbox
    bx0, by0, bx1, _ = b.bbox
    overlap = min(ax1, bx1) - max(ax0, bx0)
    narrow = min(ax1 - ax0, bx1 - bx0)
    return narrow > 0 and overlap >= 0.5 * narrow and -5 <= by0 - ay1 <= gap


def column_chains(blocks: list[Block]) -> list[list[Block]]:
    """Link each block to the nearest block directly below it in the same column."""
    nxt: dict[int, int] = {}
    has_prev: set[int] = set()
    for a in blocks:
        cands = [b for b in blocks if b.idx != a.idx and b.idx not in has_prev and _below(a, b)]
        if cands:
            b = min(cands, key=lambda c: c.bbox[1])
            nxt[a.idx] = b.idx
            has_prev.add(b.idx)
    chains = []
    for start in sorted((b for b in blocks if b.idx not in has_prev), key=lambda b: (round(b.bbox[0] / 50), b.bbox[1])):
        chain, cur = [start], start.idx
        while cur in nxt:
            cur = nxt[cur]
            chain.append(blocks[cur])
        chains.append(chain)
    return chains


def page_passages(blocks: list[Block], min_words: int, max_words: int) -> list[Passage]:
    """Merge blocks down each column chain until a passage has at least min_words."""
    def n(bs: list[Block]) -> int:
        return sum(len(x.text.split()) for x in bs)

    out: list[Passage] = []
    for chain in column_chains(blocks):
        groups: list[list[Block]] = [[]]
        for b in chain:
            cur = groups[-1]
            if cur and (n(cur) >= min_words or n(cur) + len(b.text.split()) > max_words):
                groups.append([])
            groups[-1].append(b)
        if len(groups) > 1 and n(groups[-1]) < min_words / 2:
            tail = groups.pop()
            groups[-1] += tail                      # a short tail joins the passage above it
        out += [Passage("", g[0].page_key, g) for g in groups]
    for n, p in enumerate(out):
        p.pid = f"{p.page_key}:{n:03d}"
    return out


def pdf_passages(path, pack_id: str, min_words: int, max_words: int) -> list[Passage]:
    out: list[Passage] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            out += page_passages(page_blocks(page, f"{pack_id}:{i}"), min_words, max_words)
    return out
