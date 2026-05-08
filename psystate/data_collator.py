"""Data loading + collation for PsyState.

Input: one of the ``<split>.chat.jsonl`` files produced by ``data/build_dataset.py``.
Output per batch:

* ``input_ids, attention_mask, labels, ctx_lens`` — for the LM.
* ``state_target, state_valid`` — weak state labels at previous client turn.
* ``state_next_target, state_next_valid`` — weak state labels at next client turn.
* ``strategy_target, strategy_valid`` — integer in [0, 7).
* ``uptake_target, uptake_valid`` — short-horizon outcome.
* ``quality_target, quality_valid`` — session quality (4-dim, only from KokoroChat).
* ``risk_mask`` — whether the prior client turn carried risk markers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .constants import N_STATE, N_STRATEGY, STATE_AXES, STRATEGIES

IGNORE_INDEX = -100


def _load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


class PsyChatDataset(Dataset):
    """Lazily tokenizes chat-format rows into tensor dicts."""

    def __init__(
        self,
        path: Path | str,
        tokenizer,
        max_length: int = 2048,
        max_ctx_length: int = 1800,
        limit: int = 0,
        subsample: float | int = 1.0,
        add_risk_label: bool = True,
        rag_memory=None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_ctx_length = max_ctx_length
        self.add_risk_label = add_risk_label
        self.rag_memory = rag_memory

        raw = _load_jsonl(self.path, limit=limit)
        if isinstance(subsample, float) and 0 < subsample < 1.0:
            k = int(len(raw) * subsample)
            raw = random.Random(17).sample(raw, k)
        elif isinstance(subsample, int) and subsample > 0 and subsample < len(raw):
            raw = random.Random(17).sample(raw, subsample)
        self.rows = raw

    def __len__(self) -> int:
        return len(self.rows)

    def _state_vec(self, s: dict | None) -> tuple[list[float], bool]:
        if not s:
            return [0.5] * N_STATE, False
        return [float(s.get(a, 0.5)) for a in STATE_AXES], True

    def _outcome_vec(self, o: dict | None) -> tuple[float, float, bool]:
        if not o:
            return 0.0, 0.5, False
        return float(o.get("uptake", 0.0)), float(o.get("uptake_soft", o.get("uptake", 0.5))), True

    def _quality_vec(self, q: dict | None) -> tuple[list[float], bool]:
        if not q:
            return [0.0, 0.0, 0.0, 0.0], False
        return [
            float(q.get("empathy_q", 0.0)),
            float(q.get("clarity_q", 0.0)),
            float(q.get("action_q", 0.0)),
            float(q.get("overall_q", 0.0)),
        ], True

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        meta = row.get("meta", {})
        messages: list[dict] = row["messages"]
        # Split into context (all but the final assistant turn) and reply (the final).
        assert messages[-1]["role"] == "assistant", "last turn must be counselor reply"
        ctx_msgs = messages[:-1]
        reply_text = messages[-1]["content"]

        if self.rag_memory is not None:
            augmented = self.rag_memory.augment_messages(list(messages))
            ctx_msgs = augmented[:-1]

        # Render chat template for context → ends at assistant generation cue.
        try:
            ctx_rendered = self.tokenizer.apply_chat_template(
                ctx_msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: minimal manual template.
            buf = []
            for m in ctx_msgs:
                buf.append(f"<|{m['role']}|>\n{m['content']}")
            buf.append("<|assistant|>\n")
            ctx_rendered = "".join(buf)

        ctx_ids = self.tokenizer(ctx_rendered, add_special_tokens=False, truncation=True,
                                 max_length=self.max_ctx_length)["input_ids"]
        reply_ids = self.tokenizer(reply_text + self.tokenizer.eos_token,
                                   add_special_tokens=False,
                                   truncation=True,
                                   max_length=self.max_length - len(ctx_ids))["input_ids"]

        input_ids = ctx_ids + reply_ids
        labels = [IGNORE_INDEX] * len(ctx_ids) + list(reply_ids)
        attn = [1] * len(input_ids)
        ctx_len = len(ctx_ids)

        # Targets.
        state_vec, state_valid = self._state_vec(meta.get("state"))
        state_next_vec, state_next_valid = self._state_vec(meta.get("next_state"))
        strategy_idx = meta.get("strategy_idx", -1)
        uptake, uptake_soft, uptake_valid = self._outcome_vec(meta.get("outcome_short"))
        quality_vec, quality_valid = self._quality_vec(meta.get("session_feedback"))
        risk_flags = meta.get("risk") or {}
        if isinstance(risk_flags, dict):
            risk_any = bool(risk_flags.get("any", False))
            # Back-compat: older builds may not carry "severity"; default to
            # 2 (severe) on any flagged turn so we never *silently* drop
            # risk supervision when re-using a v3-era dataset.
            if "severity" in risk_flags:
                risk_severity = int(risk_flags.get("severity", 0))
            else:
                risk_severity = 2 if risk_any else 0
        else:
            risk_any = False
            risk_severity = 0

        last_user = next((m.get("content", "") for m in reversed(ctx_msgs) if m.get("role") == "user"), "")
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
        # Measurement-quality features for the reliability gate:
        # confidence away from neutral, entropy/noisiness, utterance length, risk marker.
        confidence = torch.mean(torch.abs(state_tensor - 0.5) * 2.0).clamp(0.0, 1.0)
        entropy = torch.mean(4.0 * state_tensor * (1.0 - state_tensor)).clamp(0.0, 1.0)
        length = min(len(last_user) / 120.0, 1.0)
        meas_quality = torch.tensor(
            [float(confidence), float(entropy), float(length), float(risk_any)],
            dtype=torch.float32,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ctx_len": ctx_len,
            "state_target": state_tensor,
            "state_valid": state_valid,
            "state_next_target": torch.tensor(state_next_vec, dtype=torch.float32),
            "state_next_valid": state_next_valid,
            "strategy_target": int(strategy_idx) if strategy_idx is not None and strategy_idx >= 0 else -1,
            "strategy_valid": strategy_idx is not None and strategy_idx >= 0,
            "uptake_target": float(uptake),
            "uptake_soft_target": float(uptake_soft),
            "uptake_valid": uptake_valid,
            "measurement_quality": meas_quality,
            "quality_target": torch.tensor(quality_vec, dtype=torch.float32),
            "quality_valid": quality_valid,
            "risk_mask": risk_any,
            "risk_severity": risk_severity,
            "meta": meta,
        }


@dataclass
class PsyCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids, attn, labels, ctx_lens = [], [], [], []
        for b in batch:
            x = b["input_ids"]; a = b["attention_mask"]; l = b["labels"]
            pad = max_len - x.size(0)
            input_ids.append(torch.cat([x, x.new_full((pad,), pad_id)]))
            attn.append(torch.cat([a, a.new_zeros(pad)]))
            labels.append(torch.cat([l, l.new_full((pad,), IGNORE_INDEX)]))
            ctx_lens.append(b["ctx_len"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
            "ctx_lens": torch.tensor(ctx_lens, dtype=torch.long),
            "state_target": torch.stack([b["state_target"] for b in batch]),
            "state_valid": torch.tensor([b["state_valid"] for b in batch], dtype=torch.bool),
            "state_next_target": torch.stack([b["state_next_target"] for b in batch]),
            "state_next_valid": torch.tensor([b["state_next_valid"] for b in batch], dtype=torch.bool),
            "strategy_target": torch.tensor([b["strategy_target"] for b in batch], dtype=torch.long),
            "strategy_valid": torch.tensor([b["strategy_valid"] for b in batch], dtype=torch.bool),
            "uptake_target": torch.tensor([b["uptake_target"] for b in batch], dtype=torch.float32),
            "uptake_soft_target": torch.tensor([b["uptake_soft_target"] for b in batch], dtype=torch.float32),
            "uptake_valid": torch.tensor([b["uptake_valid"] for b in batch], dtype=torch.bool),
            "measurement_quality": torch.stack([b["measurement_quality"] for b in batch]),
            "quality_target": torch.stack([b["quality_target"] for b in batch]),
            "quality_valid": torch.tensor([b["quality_valid"] for b in batch], dtype=torch.bool),
            "risk_mask": torch.tensor([b["risk_mask"] for b in batch], dtype=torch.bool),
            "risk_severity": torch.tensor(
                [b.get("risk_severity", 0) for b in batch], dtype=torch.long,
            ),
        }
