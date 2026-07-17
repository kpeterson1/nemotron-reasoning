"""LoRA SFT training for Nemotron-3-Nano-30B-A3B.

Loads the model via the native transformers path (no `trust_remote_code` —
the repo's auto_map points at custom code that hard-imports `mamba_ssm`,
which we don't have on this aarch64 box. The native nemotron_h class falls
back to a naive Mamba implementation when kernels are missing).

Each training record in `train_file` is a dict with `text` (the full chat-
templated string) and `prompt_text` (a strict prefix of `text` ending right
before the assistant turn). We compute labels by copying input_ids and
masking the prompt portion to -100, so loss is only on the assistant
response.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def train(
    config_path: Path,
    max_steps: int | None = None,
    max_samples: int | None = None,
    smoke: bool = False,
) -> None:
    config = yaml.safe_load(config_path.read_text())

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    model_id = config["model"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] model_id={model_id}", flush=True)
    print(f"[train] output_dir={output_dir}", flush=True)

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[train] set pad_token = eos_token = {tokenizer.eos_token!r}", flush=True)

    # ---- model config: skip remote code; rely on native transformers
    # implementation. mamba_ssm + causal_conv1d are now installed, so the
    # native code's lazy-loader will activate the fused fast path. ----
    hf_config = AutoConfig.from_pretrained(model_id)
    if hasattr(hf_config, "auto_map"):
        hf_config.auto_map = {}

    print(f"[train] loading model in BF16 (~5 min)...", flush=True)
    import time
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=hf_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    print(f"[train] model loaded in {time.time()-t0:.0f}s", flush=True)

    # ---- LoRA wrap ----
    lora_kwargs: dict[str, Any] = dict(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    if config["lora"].get("target_parameters"):
        lora_kwargs["target_parameters"] = list(config["lora"]["target_parameters"])
        print(f"[train] target_parameters={lora_kwargs['target_parameters']}", flush=True)
    # Optional per-parameter rank override (3D experts can need a smaller r than 2D linears).
    if config["lora"].get("rank_pattern"):
        lora_kwargs["rank_pattern"] = dict(config["lora"]["rank_pattern"])
        print(f"[train] rank_pattern={lora_kwargs['rank_pattern']}", flush=True)
    if config["lora"].get("alpha_pattern"):
        lora_kwargs["alpha_pattern"] = dict(config["lora"]["alpha_pattern"])
    lora_config = LoraConfig(**lora_kwargs)

    # gradient checkpointing requires the input embeddings to retain grad,
    # otherwise grads don't flow back through the frozen base. Default is on
    # (memory-cheap); a config opt-out lets us speed-probe at the cost of
    # activation memory.
    grad_ckpt = bool(config["training"].get("gradient_checkpointing", True))
    print(f"[train] gradient_checkpointing={grad_ckpt}", flush=True)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Audit module/parameter matches before we proceed: this tells us instantly whether
    # target_parameters resolved to anything in the live model graph.
    targeted_params = list(getattr(model.base_model, "targeted_parameter_names", []) or [])
    targeted_mods = list(getattr(model.base_model, "targeted_module_names", []) or [])
    print(f"[train] LoRA matched {len(targeted_mods)} modules and {len(targeted_params)} parameters", flush=True)
    if targeted_params:
        for s in targeted_params[:6]:
            print(f"        param: {s}", flush=True)
        if len(targeted_params) > 6:
            print(f"        ... (+{len(targeted_params) - 6} more)", flush=True)

    # Optional warm-start: copy LoRA tensors from a previously-saved PEFT adapter
    # into the live model. Used to preserve old-v9 non-expert behavior while
    # adding routed-expert ParamWrapper coverage. Expert ParamWrapper tensors are
    # not present in the source adapter; those stay at PEFT-init values.
    #
    # Direct per-tensor assignment rather than peft.set_peft_model_state_dict,
    # because peft 0.19.1's loader calls a transformers WeightConverter helper
    # with a kwarg that doesn't exist in the installed transformers — and we
    # don't actually need its translation machinery; the rename is one regex.
    warm_start_path = config["lora"].get("warm_start_adapter")
    if warm_start_path:
        from safetensors.torch import load_file as _load_safetensors
        import re as _re
        wsp = Path(warm_start_path) / "adapter_model.safetensors"
        print(f"[warm-start] loading {wsp}", flush=True)
        src_state = _load_safetensors(str(wsp))
        print(f"[warm-start] source tensors: {len(src_state)}", flush=True)
        live_param_dict = dict(model.named_parameters())
        print(f"[warm-start] live named_parameters in model: {len(live_param_dict)}", flush=True)
        copied = 0
        skipped_shape = 0
        unbound = []
        bound_names = set()
        with torch.no_grad():
            for k, v in src_state.items():
                live_k = _re.sub(r"(lora_[AB])\.weight$", r"\1.default.weight", k)
                live = live_param_dict.get(live_k)
                if live is None:
                    unbound.append(live_k)
                    continue
                if tuple(live.shape) != tuple(v.shape):
                    skipped_shape += 1
                    if skipped_shape <= 5:
                        print(f"[warm-start] shape mismatch: {live_k}  live={tuple(live.shape)}  src={tuple(v.shape)}", flush=True)
                    continue
                live.data.copy_(v.to(dtype=live.dtype, device=live.device))
                bound_names.add(live_k)
                copied += 1
        print(f"[warm-start] copied={copied}  shape_mismatch_skipped={skipped_shape}  unbound={len(unbound)}", flush=True)
        if unbound:
            print(f"[warm-start] FATAL — {len(unbound)} src tensors had no live target; aborting", flush=True)
            for u in unbound[:10]:
                print(f"            {u}", flush=True)
            raise RuntimeError("warm_start_adapter has keys that didn't bind in the live model")
        if skipped_shape:
            print(f"[warm-start] FATAL — {skipped_shape} shape mismatches; aborting", flush=True)
            raise RuntimeError("warm_start_adapter has incompatible tensor shapes")

        if config["lora"].get("freeze_warm_start", False):
            n_frozen = 0
            for name, p in model.named_parameters():
                if name in bound_names:
                    p.requires_grad = False
                    n_frozen += 1
            print(f"[warm-start] froze {n_frozen} copied tensors (requires_grad=False)", flush=True)
            print(f"[warm-start] post-freeze trainable_parameters:", flush=True)
            model.print_trainable_parameters()

    # ---- dataset ----
    train_records = _load_jsonl(Path(config["data"]["train_file"]))
    if max_samples is not None:
        train_records = train_records[:max_samples]
        print(f"[train] SMOKE — sliced training to first {len(train_records)} records", flush=True)
    print(f"[train] loaded {len(train_records)} training records", flush=True)
    train_ds = _build_dataset(
        train_records,
        tokenizer,
        max_seq_length=config["training"]["max_seq_length"],
    )
    print(f"[train] tokenized dataset: {len(train_ds)} rows", flush=True)

    # ---- training ----
    targs_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        bf16=config["training"]["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],  # no wandb / tensorboard
        gradient_checkpointing=False,  # we already enabled it above
        remove_unused_columns=False,
    )
    if max_steps is not None:
        targs_kwargs["max_steps"] = max_steps
        targs_kwargs["save_strategy"] = "no"
        targs_kwargs["logging_steps"] = 1
        print(f"[train] DRY RUN — capping at max_steps={max_steps}", flush=True)
    targs = TrainingArguments(**targs_kwargs)

    collator = _CompletionCollator(tokenizer.pad_token_id)

    # transformers 5.x renamed `tokenizer` kwarg to `processing_class`.
    trainer_kwargs = dict(
        model=model, args=targs, train_dataset=train_ds, data_collator=collator,
    )
    import inspect as _inspect
    if "processing_class" in _inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    # Loss / grad-finiteness audit on the smoke path, before saving. This is how we
    # verify the matched 3D experts actually receive gradients.
    if smoke:
        import math
        history = getattr(trainer.state, "log_history", []) or []
        losses = [h.get("loss") for h in history if "loss" in h]
        print(f"[train] SMOKE loss trace: {losses}", flush=True)
        all_finite = all(l is not None and math.isfinite(l) for l in losses)
        print(f"[train] SMOKE loss finite: {all_finite}", flush=True)
        n_param = 0
        n_with_grad = 0
        n_nan = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            n_param += 1
            if p.grad is None:
                continue
            n_with_grad += 1
            g = p.grad
            if not torch.isfinite(g).all():
                n_nan += 1
                if n_nan <= 3:
                    print(f"        non-finite grad on {name}", flush=True)
        print(f"[train] SMOKE trainable params: {n_param}, with grad after last step: {n_with_grad}, non-finite: {n_nan}", flush=True)

    if smoke or max_steps is None:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[train] saved adapter to {output_dir}", flush=True)
    else:
        print(f"[train] DRY RUN complete — adapter not saved", flush=True)


@dataclass
class _CompletionCollator:
    """Pads input_ids/attention_mask with pad_token_id and labels with -100."""
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict:
        import torch
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for b in batch:
            ids = list(b["input_ids"])
            lbls = list(b["labels"])
            n_pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * n_pad)
            labels.append(lbls + [-100] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_dataset(records: list[dict], tokenizer, max_seq_length: int):
    """Build a tokenized dataset with completion masking. We rely on the
    pre-computed `n_tokens_prompt` from `format_traces.py` rather than
    re-tokenizing prompt_text separately, which guarantees the boundary is
    consistent with what the formatter measured."""
    from datasets import Dataset

    def tokenize(rec: dict) -> dict:
        full = tokenizer(
            rec["text"],
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        labels = list(full["input_ids"])
        # Re-tokenize prompt to find the boundary; use that as the mask cutoff
        # to be robust to small drift between the cached n_tokens_prompt and
        # the live tokenizer.
        prompt_only = tokenizer(
            rec["prompt_text"],
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        prompt_len = len(prompt_only["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        full["labels"] = labels
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    ds = Dataset.from_list(records)
    return ds.map(tokenize, remove_columns=ds.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps for dry-run validation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training records (smoke runs)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: per-step logs, grad finiteness audit, always save")
    args = parser.parse_args()
    train(
        args.config,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
