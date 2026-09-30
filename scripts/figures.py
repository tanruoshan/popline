"""Paper figures 1-4, drawn from runs/ only (no page images, no Arcanum text beyond single words).

Fig 1 effort curve (tier 1). Fig 2 article x method map. Fig 3 "Paneuropa" spellings in the OCR.
Fig 4 one page (T1-06 p.0) as passage boxes, gold IDEA article, first hit + "why" per method.
In:  runs/rq1_curves.csv, rq1_first_hits.csv, rq1_ocr.json, rq1_results.json, gold_*_bea.jsonl,
     passages.jsonl, emb_multilingual-e5-base.npz
Out: report/overleaf-popline/figures/fig_effort.pdf, fig_article_map.pdf, fig_spellings.pdf, fig_page.pdf
Run: python scripts/figures.py
"""
from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib
import numpy as np
from rapidfuzz.distance import Levenshtein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is set)
from matplotlib.patches import Rectangle  # noqa: E402

from popline import files  # noqa: E402
from popline.pdftext import tokens  # noqa: E402
from popline.retrieve import expand_fuzzy, query_terms  # noqa: E402
from popline.rq1 import rankings, tier_passages  # noqa: E402

runs, out = files.runs_dir(), files.ROOT / "report/overleaf-popline/figures"
q = files.queries()


def jl(name: str) -> list[dict]:
    return files.read_jsonl(runs / name)


def rd(name: str) -> list[dict]:
    return files.read_csv(runs / name)


# dataviz palette (validated, light mode). Method colour fixed across all figures.
INK, MUTED, GRID, NEUTRAL, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec", "#ffffff"
METHODS = [  # (name in runs, short label, colour, dash)
    ("R0 reading order", "R0 reading order", "#a3a29d", "-"),
    ("B0 keyword", "B0 paneurop*", "#2a78d6", "-"),
    ("B0-literal", "B0-literal paneuropa*", "#2a78d6", (0, (3, 1.5))),
    ("B1 lexicon exact", "B1 lexicon exact", "#eb6834", "-"),
    ("S1 lexicon fuzzy", "S1 lexicon fuzzy", "#1baf7a", "-"),
    ("S2 dense", "S2 dense", "#eda100", "-"),
    ("S3 fusion", "S3 fusion (S1+S2)", "#e87ba4", "-"),
]
SHORT = {"R0 reading order": "R0", "B0 keyword": "B0", "B0-literal": "B0-lit", "B1 lexicon exact": "B1",
         "S1 lexicon fuzzy": "S1", "S2 dense": "S2", "S3 fusion": "S3"}
COL_W = 3.33   # ACM sigconf column width, inches

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#bdbcb7", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, no Type 3 fonts (ACM TAPS)
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)   # preview only, not in git
    plt.close(fig)


# ---------------- Fig 1: reading effort ----------------
def fig_effort():
    rows = [r for r in rd("rq1_curves.csv") if r["tier"] == "1"]
    res = files.read_json(runs / "rq1_results.json")["tier1"]
    n_idea = res["idea_articles"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.axhline(0.8, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.text(1.05, 0.785, "80% of IDEA articles", color=MUTED, fontsize=6, va="top")
    for name, label, col, dash in METHODS:
        pts = [(int(r["n_read"]), int(r["found"]) / n_idea) for r in rows if r["method"] == name]
        x, y = zip(*pts)
        ax.step(x, y, where="post", color=col, lw=1.4 if name != "R0 reading order" else 1.1, ls=dash,
                label=f"{label} ({res[name]['effort_80']})", zorder=3 if name != "R0 reading order" else 2)
        ax.plot(res[name]["effort_80"], 0.8, "o", ms=3.2, color=col, mec=SURFACE, mew=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1, res["passages"])
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 5, 10, 20, 50, 100, 200, 608])
    ax.set_xticklabels(["1", "5", "10", "20", "50", "100", "200", "608"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("passages read, best first (log scale)")
    ax.set_ylabel(f"IDEA articles found (of {n_idea})")
    leg = ax.legend(loc="upper left", frameon=False, handlelength=2.2, labelspacing=0.3, columnspacing=1.2,
                    ncol=2, title="method (passages read to reach 80%, dot)", title_fontsize=6.5,
                    bbox_to_anchor=(-0.02, -0.2))
    leg._legend_box.align = "left"
    save(fig, "fig_effort")


# ---------------- Fig 2: article x method map ----------------
BINS = [(5, "1-5", "#0d366b", SURFACE), (10, "6-10", "#1c5cab", SURFACE), (20, "11-20", "#3987e5", SURFACE),
        (50, "21-50", "#86b6ef", INK), (100, "51-100", "#cde2fb", INK), (10**9, ">100", NEUTRAL, MUTED)]


def fig_article_map():
    hits = [r for r in rd("rq1_first_hits.csv") if r["tier"] == "1"]
    ocr = files.read_json(runs / "rq1_ocr.json")["tier1_idea"]["articles"]
    pos = {(r["article"], r["method"]): int(r["first_position"]) for r in hits}
    info = {r["article"]: r for r in hits}
    arts = sorted(info, key=lambda a: ocr[a]["quality"])            # worst OCR on top
    cols = [m[0] for m in METHODS]
    xs = [0, 1.25, 2.25, 3.25, 4.5, 5.5, 6.5]                          # gaps: R0 | B | S
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    for i, a in enumerate(arts):
        for x, m in zip(xs, cols):
            p = pos[(a, m)]
            _, _, fill, txt = next(b for b in BINS if p <= b[0])
            ax.add_patch(Rectangle((x + 0.04, i + 0.06), 0.92, 0.88, fc=fill, ec="none"))
            ax.text(x + 0.5, i + 0.5, str(p), ha="center", va="center", fontsize=6, color=txt)
        r = info[a]
        tag = {"PRO": "P", "NEUTRAL": "N", "CONTRA": "C", "UNSURE": "U"}[r["stance"]]
        tag += " i" if r["implicit"] == "True" else ""
        ax.text(xs[-1] + 1.15, i + 0.5, tag, ha="left", va="center", fontsize=6, color=MUTED)
    ax.set_xlim(-0.1, xs[-1] + 1.6)
    ax.set_ylim(len(arts), 0)
    ax.set_xticks([x + 0.5 for x in xs])
    ax.set_xticklabels([SHORT[m] for m in cols])
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(arts))])
    ax.set_yticklabels([f"{a.replace('T1-', '')}  {ocr[a]['quality']:.2f}" for a in arts], fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.15, -0.25, "article  OCR", ha="right", va="bottom", fontsize=6, color=MUTED, transform=ax.transData)
    ax.text(xs[-1] + 1.15, -0.25, "Gold", ha="left", va="bottom", fontsize=6, color=MUTED)
    # key below
    kx = 0.0
    for _, lab, fill, _ in BINS:
        ax.add_patch(Rectangle((kx, len(arts) + 0.55), 0.35, 0.5, fc=fill, ec="none", clip_on=False))
        ax.text(kx + 0.45, len(arts) + 0.8, lab, va="center", fontsize=6, color=MUTED)
        kx += 1.45 if lab != "51-100" else 1.7
    ax.text(0.0, len(arts) + 1.6, "cell: rank of the article's first passage. Rows: worst OCR on top.\n"
            "Gold stance: P pro, N neutral, C contra, U unsure, i implicit", fontsize=6, color=MUTED, va="top",
            linespacing=1.3)
    save(fig, "fig_article_map")


# ---------------- Fig 3: Paneuropa spellings ----------------
def paneuropa_forms(rows):
    """Tokens that look like 'Paneuropa' (first 8 letters >= 0.75 similar to 'paneurop'). Figure only."""
    return Counter(t for r in rows for t in tokens(r["text"])
                   if len(t) >= 7 and Levenshtein.normalized_similarity(t[:8], "paneurop") >= 0.75)


def fig_spellings():
    rows = jl("passages.jsonl")
    forms = paneuropa_forms(rows)
    vocab = {t for r in rows for t in tokens(r["text"])}
    fuzzy = expand_fuzzy(query_terms(q["lexicon"]), vocab, q["fuzzy_min_sim"])
    lit, pre = q["b0_literal"]["pattern"], q["b0_keyword"]["pattern"]
    classes = [
        (f'"{lit}..." (B0-literal, B0)', lambda t: t.startswith(lit), "#2a78d6"),
        (f'other "{pre}..." (B0)', lambda t: t.startswith(pre) and not t.startswith(lit), "#86b6ef"),
        ("garbled, fuzzy lexicon only (S1)", lambda t: not t.startswith(pre) and t in fuzzy, "#1baf7a"),
        ("garbled, no keyword method", lambda t: not t.startswith(pre) and t not in fuzzy, "#a3a29d"),
    ]
    total = sum(forms.values())
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for i, (label, test, col) in enumerate(classes):
        fs = sorted(((t, n) for t, n in forms.items() if test(t)), key=lambda x: (-x[1], x[0]))
        n_tok, y = sum(n for _, n in fs), i * 1.7
        x = 0.0
        for t, n in fs:
            ax.add_patch(Rectangle((x, y), n, 0.8, fc=col, ec=SURFACE, lw=0.8))
            x += n
        ax.text(0, y - 0.12, f"{label}: {n_tok} tokens ({n_tok / total:.0%}), {len(fs)} spellings",
                ha="left", va="bottom", fontsize=6, color=INK)
        ex = ", ".join(f"{t} {n}" if n > 1 else t for t, n in fs[:3])
        ax.text(x + 1.2, y + 0.4, ex, ha="left", va="center", fontsize=5.3, color=MUTED, style="italic")
    ax.set_xlim(0, 75)
    ax.set_ylim(len(classes) * 1.7 - 0.5, -0.6)
    ax.axis("off")
    ax.text(0, len(classes) * 1.7 - 0.55, "one segment = one spelling, width = tokens; right: most frequent "
            f"spellings.\nAll {len(rows)} passages, {total} tokens that look like \"Paneuropa\".",
            fontsize=5.5, color=MUTED, va="top", linespacing=1.3)
    save(fig, "fig_spellings")
    return {c[0]: sum(n for t, n in forms.items() if c[1](t)) for c in classes}, total, len(forms)


# ---------------- Fig 4: one page, what the historian sees ----------------
PAGE, ARTICLE = ("T1-06", 0), "T1-06:A1"


def tier_rankings(tier):
    """The same rankings as scripts/evaluate.py (src/popline/rq1.py), keyed by method name."""
    rows = [r for r in jl("passages.jsonl") if r["tier"] == tier]
    gold = {g["pid"] for g in jl("gold_passages_bea.jsonl")}
    rows = [r for r in rows if r["pid"] in gold]
    with np.load(runs / f"emb_{q['dense_model'].split('/')[-1]}.npz") as e:
        emb = dict(zip(e["pids"].tolist(), e["passages"]))
        concepts = e["concepts"]
    ranks = rankings(tier_passages(rows, tier, gold), q, files.config()["bm25"], emb, concepts)
    return rows, {r.method: r for r in ranks if r.kind != "baseline"}


def dot(name, col):
    """B0-literal: hollow blue dot (dashed line in Fig 1), B0: filled."""
    return {"mfc": SURFACE if name == "B0-literal" else col, "mec": col if name == "B0-literal" else INK,
            "mew": 0.9 if name == "B0-literal" else 0.3}


def fig_page():
    rows, ranks = tier_rankings(1)
    gold = {g["pid"]: g for g in jl("gold_passages_bea.jsonl")}
    first = {(r["article"], r["method"]): int(r["first_position"]) for r in rd("rq1_first_hits.csv")}
    page = [r for r in rows if (r["pack_id"], r["page"]) == PAGE]
    in_art = {r["pid"] for r in page if gold[r["pid"]]["article"] == ARTICLE}
    boxes = [b for r in page for b in r["boxes"]]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)

    fig = plt.figure(figsize=(COL_W, 2.45))
    ax = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    ax.add_patch(Rectangle((x0 - 8, y0 - 8), x1 - x0 + 16, y1 - y0 + 16, fc=SURFACE, ec="#bdbcb7", lw=0.6))
    for r in page:
        art = r["pid"] in in_art
        for b in r["boxes"]:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], lw=0.3,
                                   fc="#cde2fb" if art else NEUTRAL, ec="#1c5cab" if art else "#d6d5d0"))
    hits = []
    for name, _, col, _ in METHODS[1:]:
        rk = ranks[name]
        i, pid = next((i, p) for i, p in enumerate(rk.order, 1) if gold[p]["article"] == ARTICLE)
        assert i == first[(ARTICLE, name)], (name, i)        # same ranking as scripts/evaluate.py
        hits.append((name, col, i, pid, rk.why.get(pid, "")))
    marked = defaultdict(list)
    for name, col, i, pid, _ in hits:
        marked[pid].append((name, col))
    for pid, ms in marked.items():
        b = next(r for r in page if r["pid"] == pid)["boxes"][0]
        for j, (name, col) in enumerate(ms):
            ax.plot(b[2] + 14 + j * 34, (b[1] + b[3]) / 2, "o", ms=4.2, clip_on=False, **dot(name, col))
    ax.set_xlim(x0 - 10, x1 + 60)
    ax.set_ylim(y1 + 10, y0 - 10)
    ax.set_aspect("equal")
    ax.axis("off")

    tx = fig.add_axes([0.45, 0.0, 0.55, 1.0])
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    tx.axis("off")
    tx.text(0, 0.97, f"Gold IDEA article {ARTICLE.replace(':', ' ')} (blue boxes)", fontsize=6.5, color=INK, va="top")
    tx.text(0, 0.915, "first passage of it in each ranking, and why:", fontsize=6, color=MUTED, va="top")
    y = 0.83
    for name, col, i, pid, why in sorted(hits, key=lambda h: h[2]):
        label = next(m[1] for m in METHODS if m[0] == name)
        tx.plot(0.03, y - 0.016, "o", ms=4.2, **dot(name, col))
        tx.text(0.08, y, f"{label}: rank {i}", fontsize=6, color=INK, va="top", fontweight="bold")
        parts = why.split(" | ")[0].split(", ") if why else []
        why = (", ".join(parts[:2]) + (", ..." if len(parts) > 2 else "")) if parts else \
            "no match, reached by reading on"
        tx.text(0.08, y - 0.05, why if len(why) <= 44 else why[:42] + "...", fontsize=5.3, color=MUTED,
                va="top", style="italic")
        y -= 0.135
    save(fig, "fig_page")
    return hits


def main() -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_effort()
    fig_article_map()
    counts, total, n_forms = fig_spellings()
    hits = fig_page()
    print("spellings:", total, "tokens,", n_forms, "forms;", counts)
    print("page", ARTICLE, [(h[0], h[2]) for h in hits])
    names = ["fig_effort", "fig_article_map", "fig_spellings", "fig_page"]
    print("written:", ", ".join(f"figures/{n}.pdf" for n in names))


if __name__ == "__main__":
    main()
