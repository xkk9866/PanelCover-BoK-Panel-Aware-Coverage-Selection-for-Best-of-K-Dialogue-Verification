"""Load a trained PsyState checkpoint (LoRA adapter + head weights).

Supports both full-precision loading and QLoRA-style 4-bit loading (default),
which is useful when running eval alongside other GPU jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from psystate.model import PsyStateConfig, PsyStateModel


def load_psystate(
    ckpt_dir: str | Path,
    base_model: str,
    bf16: bool = True,
    device: str = "cuda",
    load_in_4bit: bool = True,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    cfg_path = ckpt_dir / "psystate_config.json"
    cfg_dict = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    tok = AutoTokenizer.from_pretrained(str(ckpt_dir / "adapter"), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    load_kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs.update(quantization_config=bnb_cfg, device_map={"": 0})
    else:
        load_kwargs.update(torch_dtype=compute_dtype)

    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    base = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))

    hidden = base.config.hidden_size
    cfg = PsyStateConfig(
        hidden_size=hidden,
        **{k: v for k, v in cfg_dict.items()
           if k != "hidden_size" and k in PsyStateConfig.__dataclass_fields__},
    )
    # Auto-detect v4 head presence from the saved heads.pt so old configs that
    # predate the ``use_*`` keys still load correctly.
    heads_pt = ckpt_dir / "heads.pt"
    if heads_pt.exists():
        try:
            import torch as _torch
            sd = _torch.load(heads_pt, map_location="cpu", weights_only=False)
            if any(k.startswith("risk_head.") for k in sd):
                cfg.use_safety_router = True
            if any(k.startswith("state_head.log_var_net.") for k in sd):
                cfg.use_uncertainty = True
            # v5 z-only outcome head: detect by inspecting the input
            # dimension of `outcome_head.backbone.0.weight`.  The default
            # head consumes 5+7+latent_dim (44 by default); v5 consumes 5.
            for k, v in sd.items():
                if k == "outcome_head.backbone.0.weight":
                    if v.shape[1] == cfg.n_state:
                        cfg.z_only_outcome = True
                    break
        except Exception:
            pass
    model = PsyStateModel(base, cfg)

    heads_pt = ckpt_dir / "heads.pt"
    if heads_pt.exists():
        head_state = torch.load(heads_pt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(head_state, strict=False)
        if unexpected:
            print(f"[load_ckpt] unexpected keys: {list(unexpected)[:5]}")

    # Move only auxiliary heads; backbone is already on GPU (4-bit) or handled by .to.
    if load_in_4bit:
        for name, mod in model.named_children():
            if name == "backbone":
                continue
            mod.to(device)
    else:
        model.to(device)

    model.eval()
    return model, tok, cfg
