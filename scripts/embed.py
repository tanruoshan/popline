"""S2 vectors. multilingual-e5, "query: "/"passage: " prefixes, mean pooling, L2-normalised. Needs Hugging Face.

In:  runs/passages.jsonl, configs/queries.yaml (concepts, dense_model)
Out: runs/emb_<model>.npz, runs/emb_<model>.json (model revision, versions, run time)
Run: python scripts/embed.py   (first run downloads ~1.1 GB; ~20 min on laptop CPU)
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from popline import files


def embed(texts: list[str], tok, model, batch: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)                       # mean pooling over real tokens
        out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        print(f"\r{min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return np.concatenate(out)


def main() -> None:
    q = files.queries()
    name = q["dense_model"]
    runs = files.runs_dir()
    rows = files.read_jsonl(runs / "passages.jsonl")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    torch.manual_seed(0)

    t = time.time()
    P = embed(["passage: " + r["text"] for r in rows], tok, model)
    C = embed(["query: " + c for c in q["concepts"]], tok, model)
    dst = runs / f"emb_{name.split('/')[-1]}.npz"
    np.savez_compressed(dst, pids=np.array([r["pid"] for r in rows]), passages=P, concepts=C)
    info = {"model": name, "model_revision": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "passages": len(rows), "concepts": len(C), "seconds": round(time.time() - t),
            "device": "cpu", "date": time.strftime("%Y-%m-%d %H:%M")}
    files.write_json(dst.with_suffix(".json"), info)
    print(f"{len(rows)} passages, {len(C)} concepts, {time.time() - t:.0f} s -> {dst}")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
