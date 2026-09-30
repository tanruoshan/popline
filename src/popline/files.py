"""Shared file handling for the scripts: repo paths, settings, JSONL/CSV/JSON, and a run record per script.

The run record (runs/<script>.run.json) says which code and settings made the outputs: git commit,
query version, a hash of both config files, package versions, time. Output files themselves stay unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

# Repo root: the editable install (pip install -e .) keeps the package in src/, two levels below.
ROOT = Path(os.environ.get("POPLINE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = ROOT / "configs" / "config.yaml"
QUERIES = ROOT / "configs" / "queries.yaml"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config() -> dict:
    return load_yaml(CONFIG)


def queries() -> dict:
    return load_yaml(QUERIES)


def path(key: str, cfg: dict | None = None) -> Path:
    """A path from config.yaml (section `paths`), resolved against the repo root."""
    return ROOT / (cfg or config())["paths"][key]


def runs_dir(cfg: dict | None = None) -> Path:
    d = path("runs_dir", cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, rows: Iterable[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, data: Any, ensure_ascii: bool = True) -> None:
    p.write_text(json.dumps(data, indent=2, ensure_ascii=ensure_ascii), encoding="utf-8")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))       # same hash on Windows and Linux checkouts
    return h.hexdigest()[:12]


PACKAGES = ["pymupdf", "rank-bm25", "rapidfuzz", "numpy", "pyyaml", "wordfreq", "matplotlib",
            "torch", "transformers", "openai"]


def write_run_record(script: str, outputs: list[Path], inputs: list[Path] | None = None,
                     extra: dict | None = None) -> Path:
    """runs/<script>.run.json: which code, settings and inputs made `outputs`."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    status = _git("status", "--porcelain", "--", "src", "scripts", "configs")
    record = {
        "script": f"scripts/{script}.py",
        "git_commit": _git("rev-parse", "HEAD"),
        "code_or_config_changed_since_commit": bool(status) if status is not None else None,
        "queries_version": queries().get("version"),
        "config_hash": _hash([CONFIG, QUERIES]),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): _hash([p]) for p in (inputs or []) if p.is_file()},
        "outputs": [str(p.relative_to(ROOT)).replace("\\", "/") for p in outputs],
        "python": platform.python_version(),
        "packages": versions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    dst = runs_dir() / f"{script}.run.json"
    write_json(dst, record)
    return dst
