"""Raw PDFs -> passages. Text from the raw Arcanum PDFs (OCR layer), never from annotated files.

In:  data/raw/*.pdf, data/manifest_pages.csv
Out: runs/passages.jsonl (1,047 passages incl. tier 2; not in git, holds OCR text), ingest.run.json.
     Id = <pack>:<page>:<n>
Run: python scripts/ingest.py
"""
from __future__ import annotations

import pymupdf

from popline import files
from popline.pdftext import page_blocks, page_passages


def main() -> None:
    cfg = files.config()
    raw = files.path("raw_dir", cfg)
    manifest = files.path("manifest_pages", cfg)
    out = files.runs_dir(cfg) / "passages.jsonl"
    rows = []
    for r in files.read_csv(manifest):
        with pymupdf.open(raw / r["source_file"]) as doc:
            pages = range(doc.page_count) if r["source_pages"] == "all" else [int(r["source_pages"]) - 1]
            for local, src_page in enumerate(pages):
                blocks = page_blocks(doc[src_page], f"{r['pack_id']}:{local}")
                for p in page_passages(blocks, cfg["passages"]["min_words"], cfg["passages"]["max_words"]):
                    rows.append({"pid": p.pid, "pack_id": r["pack_id"], "tier": int(r["tier"]), "page": local,
                                 "n_words": p.n_words, "boxes": [[round(v, 1) for v in b.bbox] for b in p.blocks],
                                 "text": p.text})
    files.write_jsonl(out, rows)
    files.write_run_record("ingest", [out], [manifest], {"passages": len(rows)})
    print(f"{len(rows)} passages from {raw} -> {out}")


if __name__ == "__main__":
    main()
