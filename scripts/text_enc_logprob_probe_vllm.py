"""vLLM-equivalent of scripts/text_enc_logprob_probe.py (Q9).

WHY A SEPARATE SCRIPT: the original (pulled from origin/phase2/text-enc-audit)
loads the adapter via HF `PeftModel.from_pretrained`, which fails in this env
with the transformers<->peft version bug
(`WeightConverter.__init__() got an unexpected keyword argument
'distributed_operation'`). So we teacher-force via vLLM + LoRA `prompt_logprobs`
against the CONVERTED adapter (the deployed-equivalent), exactly as
scripts/bitmanip_logprob_probe.py does. Region markers + gold trace are copied
verbatim from the original so the analysis is identical.

Teacher-forces the (v9-format, unchanged in v11) text_encryption gold trace and
reports, per trace region: mean gold logprob, divergence rate (rank>1), and the
first-divergence region. Run once per adapter (v9 vs v11bm) over the same
text_enc set, then diff.

Usage:
  python -m scripts.text_enc_logprob_probe_vllm \
    --adapter runs/train/lora_v11bm_ws300_vllm \
    --out runs/eval/text_enc_logprob_probe_v11bm.json [--restrict-failing]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V1", "0")

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEV = Path("datasets/splits/dev_frozen.jsonl")
DEFAULT_BASELINE_EVAL = Path(
    "runs/eval/dev_frozen-raw-submissions_extracted_lora_v9_warm_start_300step-1780923962.json")
HARNESS_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`")

# Region markers copied verbatim from scripts/text_enc_logprob_probe.py.
REGION_MARKERS = [
    ("intro", "This is a substitution cipher"),
    ("example_pairs", "Pair:"),
    ("example_pairs", "  "),
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


def build_full(prompt, trace, answer, tokenizer):
    messages = [{"role": "user", "content": prompt + HARNESS_SUFFIX}]
    gen_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    boxed = "\\boxed{" + answer + "}"
    if gen_prompt.rstrip().endswith("<think>"):
        completion = "\n" + trace + "\n</think>\n" + boxed
    else:
        completion = "<think>\n" + trace + "\n</think>\n" + boxed
    return gen_prompt + completion, gen_prompt


def load_targets(baseline_eval, restrict_failing):
    from src.data.text_encryption_trace import generate_trace
    rows = [json.loads(l) for l in DEV.open()
            if json.loads(l)["task_type"] == "text_encryption"]
    model_correct = None
    if restrict_failing:
        ev = json.load(Path(baseline_eval).open())
        model_correct = {p["id"] for p in ev["predictions"]
                         if p["task_type"] == "text_encryption" and p["correct"]}
    targets = []
    for r in rows:
        gen = generate_trace(r["prompt"])
        if gen is None:
            continue
        trace, _decoded = gen
        failing = (model_correct is not None) and (r["id"] not in model_correct)
        if restrict_failing and not failing:
            continue
        targets.append({"id": r["id"], "prompt": r["prompt"],
                        "answer": str(r["answer"]), "trace": trace,
                        "baseline_failing": failing})
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--baseline-eval", default=str(DEFAULT_BASELINE_EVAL))
    ap.add_argument("--restrict-failing", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    targets = load_targets(args.baseline_eval, args.restrict_failing)
    print(f"[probe] adapter={args.adapter} text_enc targets={len(targets)}", flush=True)

    llm = LLM(model=BASE, dtype="bfloat16", gpu_memory_utilization=0.85,
              enable_lora=True, max_lora_rank=32, max_loras=1,
              max_model_len=args.max_model_len, max_num_seqs=16,
              enable_prefix_caching=False, enable_chunked_prefill=True)
    tok = llm.get_tokenizer()
    lora_req = LoRARequest("probe", 1, args.adapter)

    fulls, gens = [], []
    for t in targets:
        full, gen = build_full(t["prompt"], t["trace"], t["answer"], tok)
        fulls.append(full); gens.append(gen)
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outs = llm.generate(fulls, sampling_params=sampling, lora_request=lora_req)

    results = []
    for t, gen, out in zip(targets, gens, outs):
        ptoks = out.prompt_token_ids
        plps = out.prompt_logprobs
        comp_start = len(tok(gen, add_special_tokens=False)["input_ids"])
        region_lps: dict[str, list[float]] = {}
        region_div: dict[str, int] = collections.Counter()
        region_tot: dict[str, int] = collections.Counter()
        running = ""
        first_div = None
        worst = {"logp": 1e9, "region": None, "token": None}
        for pos in range(comp_start, len(ptoks)):
            tid = ptoks[pos]
            entry = plps[pos] if pos < len(plps) else None
            if not entry or tid not in entry:
                continue
            piece = tok.decode([tid])
            running += piece
            reg = region_for_line(running.split("\n")[-1])
            lp = float(entry[tid].logprob)
            rank = entry[tid].rank
            region_lps.setdefault(reg, []).append(lp)
            region_tot[reg] += 1
            if rank != 1:
                region_div[reg] += 1
            if lp < worst["logp"]:
                worst = {"logp": round(lp, 3), "region": reg, "token": piece}
            if first_div is None and rank != 1:
                first_div = {"region": reg, "token": piece, "logp": round(lp, 3)}
        results.append({
            "id": t["id"], "baseline_failing": t["baseline_failing"],
            "region_mean_logp": {k: round(sum(v) / len(v), 3) for k, v in region_lps.items()},
            "region_div_rate": {k: round(region_div[k] / region_tot[k], 3) for k in region_tot},
            "region_tot": dict(region_tot),
            "first_divergence": first_div, "worst_token": worst,
        })

    # Aggregate per region across problems.
    agg_lp: dict[str, list[float]] = {}
    agg_div: dict[str, int] = collections.Counter()
    agg_tot: dict[str, int] = collections.Counter()
    for r in results:
        for k, v in r["region_mean_logp"].items():
            agg_lp.setdefault(k, []).append(v)
        for k, n in r["region_tot"].items():
            agg_tot[k] += n
            agg_div[k] += round(r["region_div_rate"][k] * n)
    summary = {
        "adapter": args.adapter, "n": len(results),
        "restrict_failing": args.restrict_failing,
        "region_mean_logp": {k: round(sum(v) / len(v), 3) for k, v in agg_lp.items()},
        "region_div_rate": {k: round(agg_div[k] / agg_tot[k], 4) for k in agg_tot},
        "first_div_region_hist": dict(collections.Counter(
            r["first_divergence"]["region"] if r["first_divergence"] else "NONE(clean)"
            for r in results)),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)
    order = ["intro", "example_pairs", "assembled_map", "test_decode_initial",
             "fill_unknowns", "final_decode", "other"]
    print(f"\n[probe] {args.adapter}  n={len(results)}")
    print(f"{'region':22s} {'mean_logp':>10s} {'div_rate':>9s}")
    for k in order:
        if k in summary["region_mean_logp"]:
            print(f"{k:22s} {summary['region_mean_logp'][k]:>10.3f} "
                  f"{summary['region_div_rate'].get(k,0):>9.3f}")
    print(f"first-divergence region histogram: {summary['first_div_region_hist']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
