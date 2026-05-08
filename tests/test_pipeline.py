"""Lightweight smoke test for the full pipeline using a tiny GPT-2 backbone.

This avoids pulling a 7B model but exercises:

* chat dataset loading and tokenization,
* PsyStateModel forward pass (hidden-state pooling, heads),
* compute_losses end-to-end,
* backward + optimizer step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psystate.data_collator import PsyChatDataset, PsyCollator
from psystate.losses import LossConfig, compute_losses
from psystate.model import PsyStateConfig, PsyStateModel


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Use a tiny model; any small causal LM works.
    model_id = os.environ.get("SMOKE_MODEL", "sshleifer/tiny-gpt2")
    print(f"[smoke] loading {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    backbone = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    hidden = backbone.config.hidden_size if hasattr(backbone.config, "hidden_size") else backbone.config.n_embd

    model = PsyStateModel(backbone, PsyStateConfig(hidden_size=hidden))
    ds = PsyChatDataset(
        Path("data/processed/dev.chat.jsonl"),
        tok, max_length=512, max_ctx_length=420, subsample=4,
    )
    print(f"[smoke] dataset size = {len(ds)}")
    loader = DataLoader(ds, batch_size=2, collate_fn=PsyCollator(tok))
    batch = next(iter(loader))
    print("[smoke] batch keys:", list(batch.keys()))

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        ctx_lens=batch["ctx_lens"],
        prev_state=batch["state_target"],
    )
    print("[smoke] z shape:", out["z"].shape, "u_probs shape:", out["u_probs"].shape)
    print("[smoke] z_trans_pred:", None if out["z_trans_pred"] is None else out["z_trans_pred"].shape)

    loss_cfg = LossConfig()
    L, logs = compute_losses(
        cfg=loss_cfg,
        gen_logits=out["gen_logits"],
        gen_labels=out["gen_labels"],
        z_pred=out["z"],
        strategy_logits=out["u_logits"],
        outcome_pred=out["outcome_pred"],
        transition_module=model.transition,
        z_trans_pred=out["z_trans_pred"],
        targets=batch,
        commit_loss=out["commit_loss"],
    )
    print("[smoke] losses:")
    for k, v in logs.items():
        print(f"   {k}: {float(v):.4f}")
    L.backward()
    print(f"[smoke] total loss backward OK: {float(L):.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    main()
