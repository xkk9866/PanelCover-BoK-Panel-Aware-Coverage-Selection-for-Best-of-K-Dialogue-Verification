"""DPO/IPO pilot training on BT-Greedy hard negatives.

This is the generator-level optimisation hook requested by the review:
PA-SCT-DRO should not remain only a response-cache selector.  The script trains
a LoRA adapter from DPO-ready pairs produced by
``eval.build_btgreedy_hard_negative_pairs``:

    chosen   = PA-SCT-DRO response preferred by proxy experts
    rejected = BT-Greedy response on the same context

The current dataset is intentionally small and high-confidence.  Use
``--dry_run`` first to validate the data.  A real run should be treated as a
pilot, not as a final generator, until more human BT-Greedy failure pairs are
collected.

Example
-------
python -m eval.run_dpo_hard_negatives --dry_run

python -m eval.run_dpo_hard_negatives \
  --model_name_or_path D:/models_cache/Qwen/Qwen2___5-7B-Instruct \
  --train_file data/preference/btgreedy_hard_negatives.jsonl \
  --output_dir runs/dpo_btgreedy_hardneg \
  --max_steps 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file",
                    default="data/preference/btgreedy_hard_negatives.jsonl")
    ap.add_argument("--model_name_or_path",
                    default="D:/models_cache/Qwen/Qwen2___5-7B-Instruct")
    ap.add_argument("--output_dir", default="runs/dpo_btgreedy_hardneg")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--max_prompt_length", type=int, default=768)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = _load_jsonl(Path(args.train_file))
    good = [
        r for r in rows
        if r.get("prompt") and r.get("chosen") and r.get("rejected")
        and r.get("chosen") != r.get("rejected")
    ]
    print(f"[dpo] loaded={len(rows)} valid={len(good)} from {args.train_file}")
    if good:
        ex = good[0]
        print("[dpo] example sample_id=",
              (ex.get("meta") or {}).get("sample_id", "unknown"))
        print("[dpo] prompt chars=", len(ex["prompt"]),
              "chosen chars=", len(ex["chosen"]),
              "rejected chars=", len(ex["rejected"]))
    if args.dry_run:
        return
    if not good:
        raise SystemExit("[dpo] no valid examples")

    # Imports are delayed so --dry_run can work even on machines without the
    # full training stack initialised.
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer
    import torch

    dataset = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in good
    ])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    train_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        max_length=args.max_length,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[dpo] saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
