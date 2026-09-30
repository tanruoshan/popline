"""Annotation packs (clean pages, no marks) for Bea and Simon. Not needed to run the study.

In:  data/raw/*.pdf, data/manifest_pages.csv, data/manifest_kappa.csv, seed from configs/config.yaml
Out: data/packs/original/, data/packs/original_kappa/
Run: python scripts/build_packs.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pymupdf

from popline import files

FOLDER = {"1": "tier1_must", "2": "tier2_should", "3": "tier3_control"}


def clean_copy(src: Path, dst: Path, pages: list[int] | None = None) -> None:
    """Copy PDF pages without any annotations: earlier test rectangles must not reach the annotators."""
    with pymupdf.open(src) as d:
        if pages is not None:
            d.select(pages)
        for page in d:
            for a in list(page.annots() or []):
                page.delete_annot(a)
        d.save(dst, garbage=3, deflate=True)


def main() -> None:
    cfg = files.config()
    raw, out = files.path("raw_dir", cfg), files.path("packs_dir", cfg) / "original"
    seed = cfg["seed"]
    pages = files.read_csv(files.path("manifest_pages", cfg))

    # tier 3 and kappa pages are drawn again from the seed and must match the manifests
    with pymupdf.open(raw / cfg["packs"]["unselected_run"]) as run:
        n_pages = run.page_count
    cand = [i for i in range(n_pages) if i + 1 not in cfg["packs"]["unselected_skip_pages"]]
    tier3 = sorted(random.Random(seed).sample(cand, cfg["packs"]["tier3_n_pages"]))
    assert [int(r["source_pages"]) for r in pages if r["tier"] == "3"] == [i + 1 for i in tier3], "tier 3 draw changed"

    for r in pages:
        dst = out / FOLDER[r["tier"]] / r["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if r["tier"] == "3":
            clean_copy(raw / cfg["packs"]["unselected_run"], dst, [int(r["source_pages"]) - 1])
        else:
            clean_copy(raw / r["source_file"], dst)

    units = [(r["file"], r["source_file"], k) for r in pages if r["tier"] == "1" for k in range(int(r["n_pages"]))]
    pick = sorted(random.Random(seed + 1).sample(range(len(units)), cfg["packs"]["kappa_n_pages"]))
    kappa = files.read_csv(files.path("manifest_kappa", cfg))
    assert [(units[i][0], units[i][2] + 1) for i in pick] == \
        [(k["same_as_bea_file"], int(k["page_in_that_file"])) for k in kappa], "kappa draw changed"
    for k in kappa:
        dst = out.parent / "original_kappa" / k["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        clean_copy(raw / k["source_file"], dst, [int(k["page_in_that_file"]) - 1])
    print(f"packs rebuilt in {out}; tier 3 pages {[i + 1 for i in tier3]}")


if __name__ == "__main__":
    main()
