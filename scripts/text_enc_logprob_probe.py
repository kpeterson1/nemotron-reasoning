#!/usr/bin/env python
"""Phase 2 (GPU): teacher-forcing logprob probe for text_encryption.

** REQUIRES A GPU. Do NOT run while Spark 1 owns the GPU. Queued for later. **

This is the rigorous version of the CPU structural decomposition
(scripts/text_enc_failure_decomp.py). Where the CPU proxy diffs the model's
*sampled* eval output against the solver trace, this script teacher-forces the
solver's known-correct trace through the v9 ws300 adapter and records per-token
logprobs, so we can see WHERE in the trace the model assigns low probability to
the correct next token — i.e. where it *wants* to diverge.

Motivation (from the CPU proxy, dev_frozen v9 ws300, 47 misses):
  - Solver covers 100% of text_encryption; model accuracy 43%. Pure learning gap.
  - emit==mechanical(model's own map): CORRECT 81% vs WRONG 0%. Map-faithfulness
    predicts correctness. The model abandons mechanical substitution for fluent
    English in the FINAL-DECODE region.
  This probe tests that hypothesis directly: the per-token logprob should
  collapse in the "Final decoded text" region (and at backfilled letters),
  not in the example-mapping / assembled-map regions.

Output: per-problem, mean logprob per TRACE REGION:
  intro | example_pairs | assembled_map | test_decode_initial | fill_unknowns |
  final_decode | boxed_answer
plus the single lowest-logprob token and its region.

Usage (once GPU is free):
  python scripts/text_enc_logprob_probe.py \
    --adapter runs/train/lora_v9_warm_start_300step \
    --split datasets/dev_frozen.jsonl \
    --ids-from runs/eval/text_enc_failure_decomp.json \
    --out runs/eval/text_enc_logprob_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Region tags, matched against the START of each trace line (our trace format,
# src/data/text_encryption_trace.py). A token is attributed to the region of
# the trace line it falls on.
REGION_MARKERS = [
    ("intro", "This is a substitution cipher"),
    ("example_pairs", "Pair:"),
    ("example_pairs", "  "),  # the per-word "  cw → pw (..)" lines
    ("assembled_map", "Assembled cipher"),
    ("test_decode_initial", "Decode the test cipher"),
    ("test_decode_initial", "  initial:"),
    ("fill_unknowns", "  unknown cipher letters:"),
    ("fill_unknowns", "  fill unknowns"),
    ("final_decode", "Final decoded text:"),
]


def region_for_line(line: str) -> str:
    for tag, prefix in REGION_MARKERS:
        if line.startswith(prefix):
            return tag
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/train/lora_v9_warm_start_300step")
    ap.add_argument("--split", default="datasets/dev_frozen.jsonl")
    ap.add_argument("--ids-from", default="runs/eval/text_enc_failure_decomp.json",
                    help="restrict to these problem ids (the misses); omit for all")
    ap.add_argument("--out", default="runs/eval/text_enc_logprob_probe.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from src.data.text_encryption_trace import generate_trace
    from src.data.format_training import _user_message, _assistant_text

    rows = [json.loads(l) for l in open(args.split)]
    te = [r for r in rows if r["task_type"] == "text_encryption"]

    keep_ids = None
    if args.ids_from and Path(args.ids_from).exists():
        decomp = json.load(open(args.ids_from))
        keep_ids = {r["id"] for r in decomp.get("records", [])}
        te = [r for r in te if r["id"] in keep_ids]
    if args.limit:
        te = te[: args.limit]

    print(f"[probe] {len(te)} problems; loading tokenizer + model (BF16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    if hasattr(cfg, "auto_map"):
        cfg.auto_map = {}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    results = []
    for r in te:
        gen = generate_trace(r["prompt"])
        if gen is None:
            continue
        trace, decoded = gen
        # Match the EXACT training format (src/data/format_training.py):
        # empty system turn, user = prompt + HARNESS_SUFFIX, assistant =
        # <think>..</think>\boxed{}. Render prefix and continuation separately so
        # we only score the assistant tokens.
        assistant = _assistant_text(trace, r["answer"])
        prefix_msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": _user_message(r["prompt"])},
        ]
        prefix_ids = tok.apply_chat_template(
            prefix_msgs, add_generation_prompt=True, return_tensors="pt"
        )
        cont_ids = tok(assistant, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([prefix_ids, cont_ids], dim=1).to(model.device)

        with torch.no_grad():
            logits = model(input_ids).logits
        logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = input_ids[:, 1:]
        tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # (T-1,)

        # Only score the continuation tokens.
        cont_start = prefix_ids.shape[1] - 1
        cont_logp = tok_logp[cont_start:].tolist()
        cont_token_ids = cont_ids[0].tolist()

        # Attribute each continuation token to a trace region by reconstructing
        # text incrementally and finding the current line's region.
        region_logps: dict[str, list[float]] = {}
        running = ""
        worst = {"logp": 1e9, "region": None, "token": None}
        for tid, lp in zip(cont_token_ids, cont_logp):
            piece = tok.decode([tid])
            running += piece
            cur_line = running.split("\n")[-1]
            reg = region_for_line(cur_line)
            region_logps.setdefault(reg, []).append(lp)
            if lp < worst["logp"]:
                worst = {"logp": lp, "region": reg, "token": piece}

        results.append({
            "id": r["id"],
            "gold": r["answer"],
            "region_mean_logp": {
                k: sum(v) / len(v) for k, v in region_logps.items()
            },
            "region_min_logp": {k: min(v) for k, v in region_logps.items()},
            "worst_token": worst,
        })
        print(f"[probe] {r['id']}  worst region={worst['region']} "
              f"logp={worst['logp']:.2f} tok={worst['token']!r}", flush=True)

    # Aggregate: mean of per-problem mean-logp by region.
    agg: dict[str, list[float]] = {}
    for res in results:
        for k, v in res["region_mean_logp"].items():
            agg.setdefault(k, []).append(v)
    summary = {
        "n": len(results),
        "adapter": args.adapter,
        "region_mean_logp_overall": {
            k: sum(v) / len(v) for k, v in agg.items()
        },
        "worst_region_histogram": _hist(res["worst_token"]["region"] for res in results),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("\n=== region mean logprob (less negative = model more confident) ===")
    for k, v in sorted(summary["region_mean_logp_overall"].items(),
                       key=lambda kv: kv[1]):
        print(f"  {k:24s}: {v:.3f}")
    print(f"\nworst-region histogram: {summary['worst_region_histogram']}")
    print(f"Wrote {args.out}")


def _hist(it):
    from collections import Counter
    return dict(Counter(it))


if __name__ == "__main__":
    main()
