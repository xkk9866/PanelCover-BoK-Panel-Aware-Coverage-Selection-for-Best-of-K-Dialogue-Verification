"""Shared DashScope (qwen-max) client utility for v10.

Used for:
- expert response generation;
- multi-expert judging (different system prompts treated as
  independent experts);
- red-team safety context synthesis.

The client is a thin wrapper around the OpenAI-compatible
DashScope endpoint with:

- per-call retry with exponential backoff;
- thread-safe concurrent execution (default 10 workers);
- strict JSON-mode parsing for judge / generator outputs;
- on-disk cache keyed by (model, prompt_hash) so re-runs don't
  re-hit the API.

We deliberately *do not* embed the API key in the source: it is
read from the ``DASHSCOPE_API_KEY`` env var or, as a fallback,
from a file at ``configs/dashscope.key`` (gitignored).  The
``configure_default_key`` helper installs the key supplied by the
user if neither is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "openai>=1.0 is required for DashScope (compatible-mode). "
        "Install with `pip install -U openai`."
    ) from e


_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-max"


def _read_key_from_disk() -> str | None:
    p = Path("configs/dashscope.key")
    if p.exists():
        s = p.read_text(encoding="utf-8").strip()
        if s:
            return s
    return None


def configure_default_key(key: str) -> None:
    """Persist the DashScope API key to ``configs/dashscope.key``.

    This is called once by the user-supplied bootstrap; subsequent
    runs read the key from disk (or the env var)."""
    p = Path("configs/dashscope.key")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key.strip(), encoding="utf-8")


def get_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or _read_key_from_disk()
    if not key:
        raise RuntimeError(
            "DashScope API key not found.  Set DASHSCOPE_API_KEY env "
            "var or call configure_default_key('sk-...').")
    return key


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _DiskCache:
    """Append-only JSON-line cache keyed by sha1(model+messages+kwargs).

    Multiple processes may write to the same cache file concurrently;
    we serialise writes with a process-local lock and tolerate
    duplicate keys (last write wins on read)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}
        if self.path.exists():
            # Use errors='replace' so a corrupted byte (e.g. from a
            # concurrent partial write) does not crash startup.  Each
            # line is still validated through json.loads below; lines
            # with the replacement char will fail JSON parsing and be
            # skipped silently.
            try:
                raw = self.path.read_text(encoding="utf-8",
                                            errors="replace")
            except Exception:
                raw = ""
            for line in raw.splitlines():
                if not line.strip():
                    continue
                if "\ufffd" in line:
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["val"]
                except Exception:
                    continue

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def put(self, key: str, val: Any) -> None:
        with self._lock:
            self._cache[key] = val
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "val": val},
                                   ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DashScopeClient:
    """Thread-safe DashScope chat client with concurrent dispatch."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        cache_path: str | Path | None = "data/_dashscope_cache.jsonl",
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_cap: float = 30.0,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or get_api_key()
        self.base_url = base_url
        self.model = model
        self._client = OpenAI(api_key=self.api_key, base_url=base_url,
                              timeout=timeout)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.cache: _DiskCache | None = (_DiskCache(cache_path) if cache_path
                                         else None)

    # ------------------------------------------------------------------
    @staticmethod
    def _hash(model: str, messages: list[dict], kwargs: dict) -> str:
        payload = json.dumps(
            {"m": model, "msgs": messages, "kw": kwargs},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        response_format: dict | None = None,
        seed: int | None = None,
        cache_bypass: bool = False,
    ) -> str:
        """Single (cached) chat call.  Returns the raw assistant
        message content (string)."""
        model = model or self.model
        kwargs: dict = {"temperature": float(temperature),
                        "max_tokens": int(max_tokens)}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = int(seed)

        cache_key = self._hash(model, messages, kwargs)
        if (not cache_bypass) and self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model, messages=messages, **kwargs,
                )
                content = (resp.choices[0].message.content or "").strip()
                if self.cache is not None:
                    self.cache.put(cache_key, content)
                return content
            except Exception as e:  # noqa: BLE001 — broad on purpose
                last_err = e
                wait = min(self.backoff_cap,
                           self.backoff_base ** attempt) * (
                               1.0 + 0.5 * random.random())
                time.sleep(wait)
        raise RuntimeError(
            f"DashScope chat failed after {self.max_retries} retries: "
            f"{last_err!r}")

    # ------------------------------------------------------------------
    def map_concurrent(
        self,
        items: Iterable,
        fn: Callable[[Any], Any],
        *,
        max_workers: int = 10,
        progress_every: int = 25,
    ) -> list:
        """Run ``fn`` concurrently over ``items`` and return results in
        input order.  ``fn`` must be re-entrant; results may be ``None``
        if ``fn`` returns ``None`` for a particular item.  Exceptions
        in ``fn`` are propagated."""
        items = list(items)
        out: list = [None] * len(items)
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                out[i] = fut.result()
                done += 1
                if done % progress_every == 0 or done == len(items):
                    print(f"  [dashscope] {done}/{len(items)} "
                          f"elapsed={time.time()-t0:.1f}s")
        return out


# ---------------------------------------------------------------------------
# Helpers shared across v10 modules
# ---------------------------------------------------------------------------


def parse_json_strict(s: str) -> dict | None:
    """Parse a JSON object, tolerating Markdown code fences and
    leading/trailing prose.  Returns ``None`` on failure."""
    if s is None:
        return None
    txt = s.strip()
    # Strip markdown fence
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
    txt = txt.strip()
    # Find first { ... last }
    a = txt.find("{")
    b = txt.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        return json.loads(txt[a:b + 1])
    except Exception:
        # Try to fix trailing commas and retry
        try:
            cleaned = txt[a:b + 1].replace(",}", "}").replace(",]", "]")
            return json.loads(cleaned)
        except Exception:
            return None


__all__ = [
    "DashScopeClient",
    "configure_default_key",
    "get_api_key",
    "parse_json_strict",
]
