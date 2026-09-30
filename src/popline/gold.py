"""Rectangles + notes (guideline v3) -> articles and passage labels.

Note: "A<n> <IDEA|MENTION> [PRO|CONTRA|NEUTRAL|UNSURE] [tags] [comment]". Tags: implicit, borderline,
speech, reprint, against:<X>, cont:<pack>/A<n>. Sticky note / text box "NONE" = page read, empty.
Unit = article (Rule 2): 1 rectangle per column, optional inner rectangle = IDEA part.
Passage -> article with largest covered share of its text area (>= min_share).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from popline.pdftext import Passage

NOTE = re.compile(r"^\s*A(\d+|\?)\s+(IDEA|MENTION)\b(?:\s+(PRO|CONTRA|NEUTRAL|UNSURE)\b)?\s*(.*)$", re.I | re.S)
GRADE = {"MENTION": 1, "IDEA": 2}
FLAGS = {"implicit", "borderline", "speech", "reprint"}
CONT = re.compile(r"^cont:(T\d-\d{2})/A(\d+)$", re.I)
AGAINST = re.compile(r"^against:(\S+)$", re.I)


@dataclass
class Note:
    article: str                 # "A3"; "A?" if the annotator left the number open
    grade: int                   # 1 MENTION, 2 IDEA
    stance: str | None
    flags: set[str] = field(default_factory=set)
    against: list[str] = field(default_factory=list)
    cont: str | None = None      # "T1-04:A3"
    comment: str = ""


def parse_note(text: str) -> Note | None:
    m = NOTE.match(text or "")
    if not m:
        return None
    num, grade, stance, rest = m.groups()
    note = Note("A?" if num == "?" else f"A{int(num)}", GRADE[grade.upper()], stance.upper() if stance else None)
    free = []
    for word in rest.split():
        low = word.lower()
        if low in FLAGS:
            note.flags.add(low)
        elif (c := CONT.match(word)):
            note.cont = f"{c.group(1).upper()}:A{int(c.group(2))}"
        elif (a := AGAINST.match(word)):
            note.against.append(a.group(1))
        else:
            free.append(word)
    note.comment = " ".join(free)
    return note


@dataclass
class Rect:
    page_key: str                # "<pack id>:<page index>"
    bbox: tuple[float, float, float, float]
    note: Note | None
    raw: str
    part: bool = False           # inner rectangle = IDEA part
    key: str = ""                # "A1", or "A1#2" if A1 reused

    @property
    def article_id(self) -> str | None:
        return f"{self.page_key.split(':')[0]}:{self.key}" if self.note else None


@dataclass
class Marks:
    rects: list[Rect]
    none_pages: set[str]
    problems: list[str]          # for the annotator; never guessed


def _area(b) -> float:
    return pymupdf.Rect(b).get_area()


def _inside(inner, outer) -> float:
    a = _area(inner)
    return pymupdf.Rect(inner).intersect(pymupdf.Rect(outer)).get_area() / a if a else 0.0


def read_marks(path, pack_id: str, page_offset: int = 0) -> Marks:
    """All marks in one annotated PDF. page_offset: Simon's 1-page file -> page index in Bea's file."""
    rects, none_pages, problems = [], set(), []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            key = f"{pack_id}:{i + page_offset}"
            for a in page.annots() or []:
                kind, text = a.type[1], (a.info.get("content") or "").strip()
                if kind in ("Text", "FreeText") and text.upper() == "NONE":   # sticky note or text box
                    none_pages.add(key)
                elif kind == "Square":
                    note = parse_note(text)
                    rects.append(Rect(key, tuple(a.rect), note, text))
                    if note is None:
                        w, h = a.rect.width, a.rect.height
                        tiny = " (tiny, probably drawn by accident)" if min(w, h) < 5 else ""
                        problems.append(f"{key}: rectangle without a readable note{tiny}: {text!r} "
                                        f"at {[round(v) for v in a.rect]}")
                elif text:
                    problems.append(f"{key}: {kind} annotation ignored: {text!r}")
    # same number, different note -> different articles (A1, A1#2), reported. "A?" same rule
    variants: dict[str, list[str]] = {}
    for r in rects:
        if r.note:
            notes = variants.setdefault(r.note.article, [])
            if r.raw not in notes:
                notes.append(r.raw)
            i = notes.index(r.raw)
            r.key = r.note.article if i == 0 else f"{r.note.article}#{i + 1}"
    for art, notes in variants.items():
        if len(notes) > 1 or art == "A?":
            if art == "A?" and len(notes) == 1:
                problems.append(f"{pack_id}: number left open (A?), read as one article per distinct note")
            else:
                problems.append(f"{pack_id}: article number {art} used for {len(notes)} different notes; kept apart")
    for r in rects:                  # inner = >= 90% inside a larger one, same article
        r.part = r.note is not None and any(
            o is not r and o.article_id == r.article_id and o.page_key == r.page_key
            and _area(o.bbox) > _area(r.bbox) and _inside(r.bbox, o.bbox) >= 0.9 for o in rects)
    for key in none_pages & {r.page_key for r in rects}:
        problems.append(f"{key}: page has NONE and rectangles")
    return Marks(rects, none_pages, problems)


def merge(*marks: Marks) -> Marks:
    return Marks([r for m in marks for r in m.rects], set().union(*(m.none_pages for m in marks)),
                 [p for m in marks for p in m.problems])


def unread_pages(page_keys: list[str], marks: Marks) -> list[str]:
    """Pages with neither a rectangle nor a NONE note: not read yet, so they cannot be scored."""
    seen = marks.none_pages | {r.page_key for r in marks.rects}
    return [k for k in page_keys if k not in seen]


def article_groups(marks: Marks) -> dict[str, str]:
    """Article id -> group id; an article marked cont: joins the article it continues from."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x
    for r in marks.rects:
        if r.note:
            find(r.article_id)
            if r.note.cont:
                parent[find(r.article_id)] = find(r.note.cont)
    return {a: find(a) for a in parent}


def articles(marks: Marks) -> dict[str, dict]:
    """Group id -> {grade, stance, flags, against, members}. Highest grade wins (Rule 2)."""
    groups = article_groups(marks)
    out: dict[str, dict] = {}
    for r in marks.rects:
        if not r.note:
            continue
        g = groups[r.article_id]
        a = out.setdefault(g, {"grade": 0, "stance": None, "flags": set(), "against": [], "members": set(),
                               "note": r.raw})
        a["members"].add(r.article_id)
        if r.note.grade > a["grade"]:
            a["grade"], a["stance"] = r.note.grade, r.note.stance
        a["flags"] |= r.note.flags
        a["against"] = sorted(set(a["against"]) | set(r.note.against))
    return out


def share_inside(p: Passage, bbox) -> float:
    rr = pymupdf.Rect(bbox)
    total = sum(b.area for b in p.blocks)
    return sum(pymupdf.Rect(b.bbox).intersect(rr).get_area() for b in p.blocks) / total if total else 0.0


def label_passages(passages: list[Passage], marks: Marks, min_share: float) -> dict[str, dict]:
    """pid -> {article, grade (0 = none), in_part (inside inner rectangle, or whole article if none)}."""
    groups, arts = article_groups(marks), articles(marks)
    by_page: dict[str, list[Rect]] = {}
    for r in marks.rects:
        if r.note:
            by_page.setdefault(r.page_key, []).append(r)
    has_part = {groups[r.article_id] for r in marks.rects if r.note and r.part}
    labels = {}
    for p in passages:
        cover: dict[str, float] = {}
        part_cover: dict[str, float] = {}
        for r in by_page.get(p.page_key, []):
            g = groups[r.article_id]
            s = share_inside(p, r.bbox)
            target = part_cover if r.part else cover
            target[g] = target.get(g, 0.0) + s
        best = max(cover, key=cover.get) if cover else None
        if best is None or cover[best] < min_share:
            labels[p.pid] = {"article": None, "grade": 0, "in_part": False}
            continue
        in_part = part_cover.get(best, 0.0) >= min_share if best in has_part else True
        labels[p.pid] = {"article": best, "grade": arts[best]["grade"], "in_part": in_part}
    # article smaller than any passage -> attach to the unlabelled passage holding >= half of its area
    found = {v["article"] for v in labels.values() if v["article"]}
    for g in arts:
        if g in found:
            continue
        rs = [r for r in marks.rects if r.note and not r.part and groups[r.article_id] == g]
        best, best_share = None, 0.0
        for p in passages:
            if labels[p.pid]["article"] is None and any(r.page_key == p.page_key for r in rs):
                total = sum(_area(r.bbox) for r in rs)
                inside = sum(pymupdf.Rect(b.bbox).intersect(pymupdf.Rect(r.bbox)).get_area()
                             for r in rs if r.page_key == p.page_key for b in p.blocks)
                if total and inside / total > best_share:
                    best, best_share = p.pid, inside / total
        if best and best_share >= 0.5:
            labels[best] = {"article": g, "grade": arts[g]["grade"], "in_part": True}
    return labels
