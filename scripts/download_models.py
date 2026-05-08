"""Download Qwen3-8B and Qwen2.5-14B-Instruct via ModelScope to D:/models_cache/Qwen."""
from __future__ import annotations
import sys, os
from pathlib import Path

os.environ.setdefault("MODELSCOPE_CACHE", "D:/models_cache")

from modelscope.hub.snapshot_download import snapshot_download

TARGET = Path("D:/models_cache/Qwen")
TARGET.mkdir(parents=True, exist_ok=True)

SPECS = [
    ("Qwen/Qwen3-8B",           "Qwen3-8B"),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2___5-14B-Instruct"),
]


def main() -> int:
    for repo, local in SPECS:
        dst = TARGET / local
        if dst.is_dir() and any((dst / f"model-0000{i}-of-*.safetensors").parent.glob("*.safetensors") for i in (1,)):
            # Weak check; refuse to redownload if weights already present.
            n_weights = len(list(dst.glob("*.safetensors")))
            if n_weights >= 2:
                print(f"[skip] {repo} appears present at {dst} ({n_weights} shards)")
                continue
        print(f"[download] {repo} -> {dst}", flush=True)
        path = snapshot_download(repo, cache_dir=str(TARGET.parent),
                                 ignore_file_pattern=[r".*\.gguf$", r".*\.bin$"])
        print(f"[done] {repo} downloaded to {path}", flush=True)
    print("ALL MODELS READY", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
