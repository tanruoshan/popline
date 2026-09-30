"""RQ2 input. IDEA article text from raw PDFs, cut by Bea's rectangles (borders only), cont: joined. No labels.

In:  data/packs/bea/<tier>/*.pdf, data/raw/*.pdf, data/manifest_pages.csv
Out: runs/rq2_articles.jsonl (not in git, holds OCR text), article_texts.run.json
Run: python scripts/article_texts.py
"""
from __future__ import annotations

import pymupdf

from popline import files
from popline.gold import article_groups, articles, merge, read_marks
from popline.pdftext import normalize

FOLDER = {"1": "tier1_must", "2": "tier2_should", "3": "tier3_control"}


def main() -> None:
    cfg = files.config()
    rows = files.read_csv(files.path("manifest_pages", cfg))
    by_pack = {r["pack_id"]: r for r in rows}
    order = {r["pack_id"]: i for i, r in enumerate(rows)}
    got = []
    for r in rows:
        path = files.path("bea_dir", cfg) / FOLDER[r["tier"]] / r["file"]
        if path.exists():
            got.append(read_marks(path, r["pack_id"]))
    marks = merge(*got)
    groups, arts = article_groups(marks), articles(marks)

    docs: dict[str, pymupdf.Document] = {}

    def raw_page(pack: str, local: int) -> pymupdf.Page:
        r = by_pack[pack]
        doc = docs.setdefault(pack, pymupdf.open(files.path("raw_dir", cfg) / r["source_file"]))
        return doc[local if r["source_pages"] == "all" else int(r["source_pages"]) - 1]

    def position(rect) -> tuple:           # reading order: pack, page, column (50 pt bands), top to bottom
        pack, page = rect.page_key.split(":")
        return order[pack], int(page), round(rect.bbox[0] / 50), rect.bbox[1]

    out_rows = []
    try:
        for g, a in sorted(arts.items(), key=lambda x: (order[x[0].split(":")[0]], x[0])):
            if a["grade"] != 2:
                continue
            own = [r for r in marks.rects if r.note and not r.part and groups[r.article_id] == g]
            rects = sorted(own, key=position)
            parts = []
            for r in rects:
                pack, local = r.page_key.split(":")
                parts.append(raw_page(pack, int(local)).get_textbox(pymupdf.Rect(r.bbox)))
            text = normalize("\n".join(parts))
            out_rows.append({"article": g, "tier": int(by_pack[g.split(":")[0]]["tier"]), "rectangles": len(rects),
                             "n_words": len(text.split()), "text": text})
    finally:
        for d in docs.values():
            d.close()
    out = files.runs_dir(cfg) / "rq2_articles.jsonl"
    files.write_jsonl(out, out_rows)
    files.write_run_record("article_texts", [out], [files.path("manifest_pages", cfg)], {"articles": len(out_rows)})
    print(f"{len(out_rows)} IDEA articles -> {out}")


if __name__ == "__main__":
    main()
