# Runs the study in README order (steps 1-9) and the tests. Windows PowerShell, from the repo root:
#   .\run_all.ps1              # steps 1, 2, 4, 5, 6, 8, 9 + tests (no network needed)
#   .\run_all.ps1 -Embed       # also step 3 (S2 vectors; downloads the model from Hugging Face, ~20 min CPU)
#   .\run_all.ps1 -Stance      # also step 7 (LLM run; needs .env, see .env.example)
# Needs: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; pip install -e .
# Arcanum PDFs are not in git: data\raw\ and data\packs\bea\ must be copied in by hand (see README).
param([switch]$Embed, [switch]$Stance)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Need($paths) {
    foreach ($p in $paths) {
        if (-not (Test-Path $p)) { throw "missing input: $p (see README, 'Data')" }
    }
}

function Step($n, $script, $inputs) {
    Need $inputs
    Write-Host "`n=== step $n : scripts\$script" -ForegroundColor Cyan
    python "scripts\$script"
    if ($LASTEXITCODE -ne 0) { throw "step $n ($script) failed" }
}

python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('popline') is None)"
if ($LASTEXITCODE -ne 0) { throw "package not installed: run 'pip install -e .' in the repo root" }

Step 1 ingest.py        @("data\raw", "data\manifest_pages.csv")
Step 2 gold_report.py   @("data\packs\bea", "runs\passages.jsonl")
if ($Embed) { Step 3 embed.py @("runs\passages.jsonl") }
Step 4 evaluate.py      @("runs\passages.jsonl", "runs\gold_articles_bea.jsonl", "runs\emb_multilingual-e5-base.npz")
Step 5 ocr_quality.py   @("runs\passages.jsonl", "runs\gold_passages_bea.jsonl", "runs\rq1_first_hits.csv")
Step 6 article_texts.py @("data\packs\bea", "data\raw")
if ($Stance) { Step 7 stance.py @("runs\rq2_articles.jsonl", ".env") }
Step 8 evaluate_stance.py @("runs\rq2_articles.jsonl", "runs\gold_articles_bea.jsonl", "runs\rq2_stance_qwen3-30b-a3b-instruct-2507.jsonl")
Step 9 figures.py       @("runs\rq1_results.json", "runs\rq1_ocr.json")

Write-Host "`n=== tests" -ForegroundColor Cyan
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }
Write-Host "`ndone. Run records (code, config, inputs of each step): runs\*.run.json" -ForegroundColor Green
