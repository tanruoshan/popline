# Popline

**Finding the European idea in noisy interwar newspapers: retrieval and stance with historians in the loop.**
Group Project in Human-Centred AI, University of Regensburg, SS 2026.
Team: Ruo Shan Tan, Simon Manzenberger, Bea Dippold.

Historians search digitised newspaper archives by keyword, download the matching pages, and then read each page to find the article that really discusses their topic. On old Fraktur OCR this is slow, and keywords miss misspelled words (`Panzuropa`, `baneuropa`, `bölferbund`). This project measures how well a machine can do that last step: ranking the passages on a page that discuss the idea of European unity (Paneuropa, 1926 and 1929), and reading their stance with a reason and a quote a historian can check.

| Document | What it is |
|---|---|
| [`runs/`](runs/) | results: `rq1_results.json` (Tables 2-4), `rq1_ocr.json`, `rq2_results.json` + `rq2_table.csv` (Table 5) |
| [`docs/annotation_guideline.md`](docs/annotation_guideline.md) | how the gold standard is marked (v3) |

## Setup (once)

Python 3.11 or newer (the study ran on 3.13, Windows). Windows PowerShell, from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[figures,dev]"     # the package popline (src/) + what steps 1, 2, 4-6, 8, 9 and the tests need
python -m pytest -q                 # 27 tests, no data needed
```

Exact versions of the study run, including `torch`/`transformers` for step 3 and `openai` for step 7: `pip install -r requirements.txt` (then `pip install -e .` again).

Raw data: copy `..\paneuropa\*.pdf` + `..\BanaterDeutscheZeitung_1929_08-1708646825__pages1-50.pdf` into `data\raw\` (not in git).
Annotated packs: `data\packs\bea\tier1_must\` (v2), `data\packs\bea\tier3_control\` (not in git).

## Pipeline (run order)

All at once: `.\run_all.ps1` (steps 1, 2, 4, 5, 6, 8, 9 and the tests; add `-Embed` for step 3, `-Stance` for step 7). It checks each step's inputs first.
Text always comes from the raw PDFs (OCR layer); the annotated files only give rectangles = gold standard.
Every step also writes `runs\<script>.run.json`: git commit, query version, config hash, input hashes, package versions.

| # | Run (`python ...`) | In | Out |
|---|---|---|---|
| 1 | `scripts\ingest.py` | `data\raw\`, `data\manifest_pages.csv` | `runs\passages.jsonl` (1,047 passages incl. the dropped tier 2; 896 in tiers 1 and 3 are scored; not in git) |
| 2 | `scripts\gold_report.py` | `data\packs\bea\`, passages | `runs\gold_articles_<annotator>.jsonl`, `runs\gold_passages_<annotator>.jsonl` + check list |
| 3 | `scripts\embed.py` | passages, `configs\queries.yaml` | `runs\emb_multilingual-e5-base.npz` / `.json` (~20 min CPU, needs Hugging Face) |
| 4 | `scripts\evaluate.py` | passages, gold, embeddings | `runs\rq1_results.json`, `rq1_first_hits.csv`, `rq1_curves.csv`, `rq1_top10.jsonl` |
| 5 | `scripts\ocr_quality.py` | passages, gold, `rq1_first_hits.csv` | `runs\ocr_quality.csv`, `runs\rq1_ocr.json` |
| 6 | `scripts\article_texts.py` | `data\packs\bea\`, `data\raw\` | `runs\rq2_articles.jsonl` (16 IDEA articles, not in git) |
| 7 | `scripts\stance.py` | `runs\rq2_articles.jsonl`, `.env` (see `.env.example`) | `runs\rq2_stance_<model>.jsonl` (`--dry-run` = show prompt) |
| 8 | `scripts\evaluate_stance.py` | stance, gold, articles | `runs\rq2_results.json`, `runs\rq2_table.csv` |
| 9 | `scripts\figures.py` | `runs\` (steps 1-5), `configs\queries.yaml` | `report\overleaf-popline\figures\fig_effort.pdf`, `fig_article_map.pdf`, `fig_spellings.pdf`, `fig_page.pdf` (+ `.png` previews, not in git) |
| - | `python -m pytest -q` | | 27 tests, incl. one end-to-end run of RQ1 on a synthetic mini corpus |
| - | `ruff check .` | | lint (settings in `pyproject.toml`) |
| - | `scripts\build_packs.py` | `data\raw\` | clean annotation packs; not needed to run the study |

Settings: `configs\config.yaml` (paths, passage cut, BM25 k1/b, evaluation) and `configs\queries.yaml` (word list, concepts, fusion constant). Change a value there, never in code.

### RQ1 (retrieval). `scripts\evaluate.py` + `src\popline\rq1.py`
Queries in `configs\queries.yaml`: version q1 was pre-registered on 29 Sep before any label came back; q1b only adds B0-literal after the first results.
- R0 reading order: no ranking, pages top to bottom. Out: `rq1_results.json` -> "R0 reading order".
- B0 keyword: "paneurop*" hits (`retrieve.keyword_b0`). Out: "B0 keyword".
- B0-literal: "paneuropa*" only. Added after first results, not pre-registered. Out: "B0-literal".
- B1 lexicon exact: word list, exact spelling, BM25 (`retrieve.bm25_expanded`; k1 1.5, b 0.75, IDF per tier). Out: "B1 lexicon exact".
- S1 lexicon fuzzy: same list, OCR-tolerant (Levenshtein 0.8), BM25. Out: "S1 lexicon fuzzy".
- S2 dense: e5 vectors (step 3), closest concept (`retrieve.dense`). Out: "S2 dense".
- S3 fusion: S1 + S2, reciprocal rank fusion (`retrieve.rrf`). Out: "S3 fusion".
- Scoring unit = article: found when any of its passages is read. Main: passages read to 80% / 100% of IDEA articles.
- Per article first hit: `rq1_first_hits.csv`. Curves: `rq1_curves.csv`. Top 10 + why: `rq1_top10.jsonl`.
- By OCR quality (step 5): `rq1_ocr.json`.
- Figures (step 9): effort curve, article x method map, "Paneuropa" spellings, one page (T1-06).

### RQ2 (stance with evidence)
- Model `qwen3-30b-a3b-instruct-2507` on GWDG SAIA, temperature 0. Why: open weights, so the run can be repeated with the same model; GWDG hosts it on its own hardware and states that prompts and replies are not stored and not used for training (Arcanum text). Prompt: `src\popline\stance.py` (guideline rule word for word).
- Steps 6 to 8. Out: `rq2_results.json` (confusion, agreement, kappa, quote check), `rq2_table.csv` (per article).

### Agreement between annotators
- Dropped 30 Sep (no second historian in time; limitation in the paper). `gold_report.py` still reads `data\packs\simon\` if pages ever arrive; `data\manifest_kappa.csv` unused.

## Data

Arcanum pages (subscription) and annotated PDFs are **not in this repo**. `data\manifest_pages.csv` lists every page + source file (`data\manifest_kappa.csv`: unused, agreement dropped). Derived labels, vectors and results: `runs\`.
