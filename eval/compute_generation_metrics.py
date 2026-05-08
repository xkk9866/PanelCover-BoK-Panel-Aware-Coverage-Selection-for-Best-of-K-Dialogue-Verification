r"""Compute standard generation-quality metrics on cached fair Best-of-N
responses against the original PsyDial gold next-turn assistant message.

Metrics (character-level for Mandarin, following ESConv/CPsyCoun/PsyDial
conventions used by MISC, MultiESC, TransESC, EmoDynamiX):

  - BLEU-1, BLEU-2, BLEU-3, BLEU-4 (corpus BLEU with smoothing-1)
  - ROUGE-L (LCS-based F1)
  - Distinct-1, Distinct-2 (lexical diversity)
  - mean character length

Inputs
------
  - data/fair_bon_v12/{system}_bon_responses.jsonl
       schema: {"sample_id", "system_id", "selected_strategy", "response", ...}
  - data/judge_eval_v10/v10_eval_contexts.jsonl  (sample_id -> dialog_id+turn_idx)
  - data/raw/psydial-{d0_m,d1,d2,d3,d4}/PsyDial-D*.json  (gold assistant turn)

Outputs
-------
  - results/generation_metrics.json
  - prints summary table for inspection

Usage
-----
    python -m eval.compute_generation_metrics
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_CTX = REPO / "data/judge_eval_v10/v10_eval_contexts.jsonl"
RAW_DIR = REPO / "data/raw"
RESP_DIR = REPO / "data/fair_bon_v12"
OUT = REPO / "results/generation_metrics.json"

SYSTEMS: list[tuple[str, str]] = [
    ("pasct_plus",     "pasct_plus_bon_responses.jsonl"),
    ("pasct_dro_anchor", "pasct_dro_anchor_bon_responses.jsonl"),
    ("psystate_pasct", "psystate_pasct_bon_responses.jsonl"),
    ("psystate_sctbok", "psystate_sctbok_bon_responses.jsonl"),
    ("oracle",         "oracle_bon_responses.jsonl"),
    ("misc",           "misc_bon_responses.jsonl"),
    ("multiesc",       "multiesc_bon_responses.jsonl"),
    ("transesc",       "transesc_bon_responses.jsonl"),
    ("kemi",           "kemi_bon_responses.jsonl"),
    ("rag",            "rag_bon_responses.jsonl"),
    ("majority",       "majority_bon_responses.jsonl"),
    ("lexicon",        "lexicon_bon_responses.jsonl"),
]

PSYDIAL_VARIANTS = ["d0_m", "d1", "d2", "d3", "d4", "d101"]


def _norm(s: str) -> str:
    """Strip whitespace; keep all printable characters incl. Chinese punct."""
    return re.sub(r"\s+", "", s or "")


def _chars(s: str) -> list[str]:
    return list(_norm(s))


def _ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _modified_precision(hyp: list[str], ref: list[str], n: int) -> tuple[int, int]:
    if len(hyp) < n:
        return 0, 0
    h = Counter(_ngrams(hyp, n))
    r = Counter(_ngrams(ref, n))
    overlap = sum(min(h[g], r[g]) for g in h)
    total = max(1, sum(h.values()))
    return overlap, total


def _bleu_corpus(hyps: list[list[str]], refs: list[list[str]], max_n: int = 4) -> dict[str, float]:
    """Char-level corpus BLEU with smoothing method 1 (add 1 to numerator if zero)."""
    sums_num = [0] * max_n
    sums_den = [0] * max_n
    sum_h_len = 0
    sum_r_len = 0
    for h, r in zip(hyps, refs):
        if not h or not r:
            continue
        sum_h_len += len(h)
        sum_r_len += len(r)
        for n in range(1, max_n + 1):
            num, den = _modified_precision(h, r, n)
            sums_num[n - 1] += num
            sums_den[n - 1] += den
    bps = []
    for n in range(1, max_n + 1):
        num = sums_num[n - 1]
        den = sums_den[n - 1] or 1
        if num == 0:
            num = 1
            den += 1
        bps.append(num / den)
    out: dict[str, float] = {}
    for k in range(1, max_n + 1):
        log_p = sum(math.log(p) for p in bps[:k]) / k
        bp = 1.0
        if sum_h_len > 0 and sum_h_len < sum_r_len:
            bp = math.exp(1 - sum_r_len / sum_h_len)
        out[f"BLEU-{k}"] = bp * math.exp(log_p)
    return out


def _rouge_l(hyp: list[str], ref: list[str], beta: float = 1.2) -> float:
    """LCS-based ROUGE-L F-measure (beta=1.2 follows MISC/MultiESC conventions)."""
    if not hyp or not ref:
        return 0.0
    n, m = len(hyp), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = hyp[i - 1]
        for j in range(1, m + 1):
            if ai == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    return ((1 + beta**2) * p * r) / (r + (beta**2) * p)


def _distinct(corpus_chars: list[list[str]], n: int) -> float:
    seen: set[tuple[str, ...]] = set()
    total = 0
    for toks in corpus_chars:
        if len(toks) < n:
            continue
        for g in _ngrams(toks, n):
            seen.add(g)
            total += 1
    if total == 0:
        return 0.0
    return len(seen) / total


def _build_gold_index() -> dict[str, str]:
    """Map dialog_id -> {turn_idx: gold_assistant_text}."""
    idx: dict[str, dict[int, str]] = {}
    for variant in PSYDIAL_VARIANTS:
        if variant == "d101":
            fp = RAW_DIR / "psydial-d101" / "PsyDial-D101.json"
            if not fp.exists():
                continue
            with fp.open(encoding="utf-8") as f:
                raw = json.load(f)
            for rec in raw:
                msgs = rec.get("messages", [])
                turns = [m for m in msgs if m.get("role") in {"user", "assistant"}]
                first_user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                from data.build_dataset import stable_hash  # type: ignore
                did = f"psydial-d101-{stable_hash(first_user):08x}"
                slot = idx.setdefault(did, {})
                for i, t in enumerate(turns):
                    if t.get("role") == "assistant":
                        slot[i] = t.get("content", "")
            continue
        fname = {
            "d0_m": "PsyDial-D0_m.json",
            "d1":   "PsyDial-D1.json",
            "d2":   "PsyDial-D2.json",
            "d3":   "PsyDial-D3.json",
            "d4":   "PsyDial-D4.json",
        }[variant]
        fp = RAW_DIR / f"psydial-{variant}" / fname
        if not fp.exists():
            continue
        with fp.open(encoding="utf-8") as f:
            raw = json.load(f)
        for i, rec in enumerate(raw):
            msgs = rec.get("messages", [])
            turns = [m for m in msgs if m.get("role") in {"user", "assistant"}]
            did = f"psydial-{variant}-{i:05d}"
            slot = idx.setdefault(did, {})
            for j, t in enumerate(turns):
                if t.get("role") == "assistant":
                    slot[j] = t.get("content", "")
    flat: dict[str, str] = {}
    for did, slot in idx.items():
        for ti, txt in slot.items():
            flat[f"{did}::{ti}"] = txt
    return flat


def _load_gold_for_eval() -> dict[str, str]:
    """Map sample_id -> gold reference (next assistant turn)."""
    flat = _build_gold_index()
    out: dict[str, str] = {}
    n_missing = 0
    with EVAL_CTX.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            did = row["dialog_id"]
            ti = int(row["turn_idx"])
            ref = flat.get(f"{did}::{ti}")
            if ref is None:
                n_missing += 1
                continue
            out[sid] = ref
    if n_missing:
        print(f"[gen-metrics] WARN: {n_missing} eval rows have no gold reference")
    return out


def _load_responses(name: str, fname: str) -> dict[str, str]:
    fp = RESP_DIR / fname
    out: dict[str, str] = {}
    with fp.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["sample_id"]] = row.get("response", "")
    return out


def _system_metrics(hyps_by_sid: dict[str, str], golds: dict[str, str]) -> dict[str, float]:
    sids = sorted(set(hyps_by_sid) & set(golds))
    hyps_chars = [_chars(hyps_by_sid[s]) for s in sids]
    refs_chars = [_chars(golds[s]) for s in sids]
    lengths = [len(c) for c in hyps_chars]
    bleu = _bleu_corpus(hyps_chars, refs_chars)
    rouges = [
        _rouge_l(h, r) for h, r in zip(hyps_chars, refs_chars)
        if h and r
    ]
    rouge_l = sum(rouges) / max(1, len(rouges))
    d1 = _distinct(hyps_chars, 1)
    d2 = _distinct(hyps_chars, 2)
    return {
        "n": len(sids),
        **bleu,
        "ROUGE-L": rouge_l,
        "Distinct-1": d1,
        "Distinct-2": d2,
        "mean_chars": sum(lengths) / max(1, len(lengths)),
    }


def main() -> None:
    print("[gen-metrics] building gold index from PsyDial raw JSONs...")
    golds = _load_gold_for_eval()
    print(f"[gen-metrics] {len(golds)} sample_ids with gold references")

    out: dict[str, dict] = {"systems": {}, "n_eval": len(golds)}
    for name, fname in SYSTEMS:
        try:
            hyps = _load_responses(name, fname)
        except FileNotFoundError:
            print(f"[gen-metrics] missing {fname}, skipping {name}")
            continue
        m = _system_metrics(hyps, golds)
        out["systems"][name] = m
        print(
            f"[gen-metrics] {name:<14s} | n={m['n']:3d} | "
            f"B1={m['BLEU-1']:.4f} B2={m['BLEU-2']:.4f} "
            f"B3={m['BLEU-3']:.4f} B4={m['BLEU-4']:.4f} | "
            f"R-L={m['ROUGE-L']:.4f} | "
            f"D1={m['Distinct-1']:.4f} D2={m['Distinct-2']:.4f} | "
            f"len={m['mean_chars']:.1f}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gen-metrics] wrote {OUT}")


if __name__ == "__main__":
    main()
