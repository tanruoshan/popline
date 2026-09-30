"""Annotators' rectangles -> gold standard + list of things to check. PDFs never changed.

In:  data/packs/bea/<tier>/*.pdf, data/packs/simon/kappa_pages/*.pdf (if any), runs/passages.jsonl
Out: runs/gold_articles_<annotator>.jsonl, runs/gold_passages_<annotator>.jsonl, gold_report.run.json,
     report on screen
Run: python scripts/gold_report.py
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

from popline import files
from popline.gold import Marks, articles, label_passages, merge, read_marks, unread_pages
from popline.pdftext import Block, Passage

FOLDER = {"1": "tier1_must", "2": "tier2_should", "3": "tier3_control"}


def load_bea(cfg: dict, pages: list[dict]) -> tuple[Marks | None, list[str]]:
    got, keys = [], []
    for r in pages:
        path = files.path("bea_dir", cfg) / FOLDER[r["tier"]] / r["file"]
        if path.exists():
            got.append(read_marks(path, r["pack_id"]))
            keys += [f"{r['pack_id']}:{i}" for i in range(int(r["n_pages"]))]
    return (merge(*got) if got else None), keys


def load_simon(cfg: dict, pages: list[dict]) -> tuple[Marks | None, list[str]]:
    got, keys = [], []
    for k in files.read_csv(files.path("manifest_kappa", cfg)):
        path = files.path("simon_dir", cfg) / "kappa_pages" / k["file"]
        if path.exists():
            pack = k["same_as_bea_file"].split("_")[0]
            got.append(read_marks(path, pack, page_offset=int(k["page_in_that_file"]) - 1))
            keys.append(f"{pack}:{int(k['page_in_that_file']) - 1}")
    return (merge(*got) if got else None), keys


def report(name, marks, keys, passages, text_of, tier_of, in_rect_share):
    print(f"\n===== {name}: {len(keys)} pages returned =====")
    unread = unread_pages(keys, marks)
    arts = articles(marks)
    in_scope = [p for p in passages if p.page_key in set(keys)]
    labels = label_passages(in_scope, marks, in_rect_share)
    hit = collections.Counter(v["article"] for v in labels.values() if v["article"])
    for tier in sorted({tier_of[k.split(":")[0]] for k in keys}):
        tk = [k for k in keys if tier_of[k.split(":")[0]] == tier]
        tier_packs = {k.split(":")[0] for k in tk}
        ta = {g: a for g, a in arts.items() if tier_of[g.split(":")[0]] == tier}
        grades = collections.Counter("IDEA" if a["grade"] == 2 else "MENTION" for a in ta.values())
        stances = collections.Counter(a["stance"] for a in ta.values() if a["grade"] == 2)
        flags = collections.Counter(f for a in ta.values() for f in a["flags"])
        empty = len([k for k in tk if k in marks.none_pages])
        print(f"tier {tier}: {len(tk)} pages, {empty} marked NONE, {len([k for k in tk if k in unread])} not read yet")
        print(f"  articles: {dict(grades)}; IDEA stance: {dict(stances)}; tags: {dict(flags)}")
        print(f"  passages in tier: {len([p for p in in_scope if p.page_key in set(tk)])}, "
              f"of which in an IDEA article: "
              f"{len([p for p, v in labels.items() if v['grade'] == 2 and p.split(':')[0] in tier_packs])}")
    no_passage = [g for g in arts if g not in hit]
    if unread:
        print("NOT READ YET (no rectangle and no NONE):", ", ".join(unread))
    if no_passage:
        print("ARTICLES WITHOUT ANY PASSAGE (rectangle too small for the passage cut):", ", ".join(no_passage))
    kw = re.compile(r"europ|paneurop", re.I)
    for g, a in sorted(arts.items()):
        if a["grade"] == 2:
            words = " ".join(text_of[p] for p, v in labels.items() if v["article"] == g)
            has = bool(kw.search(words))
            if has == ("implicit" in a["flags"]):
                what = "tagged implicit but its OCR text contains Europ*" if has else \
                    "not tagged implicit and its OCR text has no Europ*"
                print(f"  check implicit: {g} is {what}")
    if marks.problems:
        print("TO CHECK WITH THE ANNOTATOR:")
        for p in marks.problems:
            print("  -", p)
    return arts, labels


def save(out: Path, name: str, arts: dict, labels: dict, tier_of: dict) -> list[Path]:
    art_file, pas_file = out / f"gold_articles_{name}.jsonl", out / f"gold_passages_{name}.jsonl"
    files.write_jsonl(art_file, ({"article": g, "tier": tier_of[g.split(":")[0]], "grade": a["grade"],
                                  "stance": a["stance"], "flags": sorted(a["flags"]), "against": a["against"],
                                  "members": sorted(a["members"]), "note": a["note"]} for g, a in sorted(arts.items())))
    files.write_jsonl(pas_file, ({"pid": pid, **v} for pid, v in labels.items()))
    return [art_file, pas_file]


def main() -> None:
    cfg = files.config()
    out = files.runs_dir(cfg)
    pages = files.read_csv(files.path("manifest_pages", cfg))
    tier_of = {r["pack_id"]: int(r["tier"]) for r in pages}
    rows = files.read_jsonl(out / "passages.jsonl")
    passages = []
    for r in rows:
        key = f"{r['pack_id']}:{r['page']}"
        passages.append(Passage(r["pid"], key, [Block(key, i, tuple(b), "") for i, b in enumerate(r["boxes"])]))
    text_of = {r["pid"]: r["text"] for r in rows}

    written = []
    for name, loader in [("bea", load_bea), ("simon", load_simon)]:
        marks, keys = loader(cfg, pages)
        if marks is None:
            print(f"\n===== {name}: nothing returned yet =====")
            continue
        arts, labels = report(name, marks, keys, passages, text_of, tier_of, cfg["passages"]["in_rect_share"])
        written += save(out, name, arts, labels, tier_of)
    files.write_run_record("gold_report", written, [out / "passages.jsonl", files.path("manifest_pages", cfg)])


if __name__ == "__main__":
    main()
