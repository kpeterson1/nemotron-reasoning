"""Teacher-forced per-token logprob probe for bit_manip traces (Q8).

Uses vLLM + LoRA `prompt_logprobs` to teacher-force a bit_manipulation gold
trace at the inference prompt shape and record, per completion token, the gold
logprob and rank (rank==1 => model argmax == gold). Localizes the first
divergence (rank>1) by structural trace region — i.e. WHERE the model chokes.

Generalized to compare v9 (v4 trace) vs v11bm (v5 trace):
  --adapter        vLLM-converted LoRA dir to probe
  --trace-version  v4 | v5  (which gold trace to teacher-force)
  --baseline-eval  v9 eval JSON; defines the "previously-failing" set
                   (= v4-covered AND model-wrong under v9)
  --restrict-failing  probe only those previously-failing IDs
  --out            output JSON

The "rule region" (analog of where v9 chokes) is RULE_STATEMENT for v4 and
RULE_DERIVATION for v5; the summary reports how many of the probed set first
diverge there.

Usage (v9 baseline, reproduce original):
  python -m scripts.bitmanip_logprob_probe --out runs/eval/bitmanip_logprob_probe.json
Usage (v11bm, restricted to the 47 previously-failing):
  python -m scripts.bitmanip_logprob_probe \
    --adapter runs/train/lora_v11bm_ws300_vllm --trace-version v5 \
    --restrict-failing --out runs/eval/bitmanip_logprob_probe_v11bm.json
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
DEFAULT_ADAPTER = "submissions/extracted/lora_v9_warm_start_300step"
DEV = Path("datasets/splits/dev_frozen.jsonl")
DEFAULT_BASELINE_EVAL = Path(
    "runs/eval/dev_frozen-raw-submissions_extracted_lora_v9_warm_start_300step-1780923962.json")
HARNESS_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`")

REGION_MARKERS = {
    "v4": [
        ("The test input is", "intro"),
        ("Examples:", "examples"),
        ("After trying candidate operations", "RULE_STATEMENT"),
        ("check: rule(", "check"),
        ("Apply to the test", "APPLY_ANSWER"),
        ("</think>", "think_close"),
        ("\\boxed{", "boxed"),
    ],
    "v5": [
        ("Test input:", "intro"),
        ("Examples (input", "examples"),
        ("Input columns:", "columns"),
        ("Bit 0:", "RULE_DERIVATION"),
        ("Apply to test", "APPLY_ANSWER"),
        ("\nfinal ", "final"),
    ],
}
REGION_MARKERS["v5cut1"] = REGION_MARKERS["v5"]  # cut1 just omits the examples block
RULE_REGION = {"v4": "RULE_STATEMENT", "v5": "RULE_DERIVATION", "v5cut1": "RULE_DERIVATION"}


def region_boundaries(text, comp_start, markers):
    bounds = []
    for marker, region in markers:
        idx = text.find(marker, comp_start)
        if idx != -1:
            bounds.append((idx, region))
    bounds.sort()
    return bounds


def region_at(offset, bounds):
    region = "think_open"
    for off, r in bounds:
        if offset >= off:
            region = r
        else:
            break
    return region


def _trace_fn(version):
    if version == "v4":
        from src.data.bit_manip_trace_v4 import generate_trace
    elif version == "v5cut1":
        from src.data.bit_manip_trace_v5_cut1 import generate_trace
    else:
        from src.data.bit_manip_trace_v5 import generate_trace
    return generate_trace


def load_targets(trace_version, baseline_eval, restrict_failing):
    """Targets = v4-covered bit_manip. `failing` flag = model-wrong under the
    baseline eval (defines the 47). The trace probed is `trace_version`'s gold
    (verified answer==gt); rows the chosen version can't derive are dropped."""
    from src.data.bit_manip_solver_v3 import parse_examples as bmp, solve as bms
    from src.data.bit_manip_trace_v4 import generate_trace as v4gen
    gen = _trace_fn(trace_version)
    rows = [json.loads(l) for l in DEV.open()
            if json.loads(l)["task_type"] == "bit_manipulation"]
    ev = json.load(Path(baseline_eval).open())
    model_correct = {p["id"] for p in ev["predictions"]
                     if p["task_type"] == "bit_manipulation" and p["correct"]}
    targets, skipped = [], []
    for r in rows:
        pairs, q = bmp(r["prompt"])
        if not pairs or q is None:
            continue
        rule = bms(pairs)
        if rule is None:
            continue
        v4res = v4gen(rule, pairs, q)               # defines the covered set
        if v4res is None or v4res[1] != str(r["answer"]).strip():
            continue
        failing = r["id"] not in model_correct
        if restrict_failing and not failing:
            continue
        res = gen(rule, pairs, q)                    # the trace we probe
        if res is None or res[1] != str(r["answer"]).strip():
            skipped.append(r["id"])                  # chosen version can't derive
            continue
        targets.append({
            "id": r["id"], "prompt": r["prompt"], "answer": str(r["answer"]),
            "trace": res[0], "baseline_failing": failing,
        })
    return targets, skipped


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--trace-version", choices=["v4", "v5", "v5cut1"], default="v4")
    ap.add_argument("--baseline-eval", default=str(DEFAULT_BASELINE_EVAL))
    ap.add_argument("--restrict-failing", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    markers = REGION_MARKERS[args.trace_version]
    rule_region = RULE_REGION[args.trace_version]
    targets, skipped = load_targets(args.trace_version, args.baseline_eval,
                                    args.restrict_failing)
    print(f"[probe] adapter={args.adapter} trace={args.trace_version} "
          f"targets={len(targets)} skipped(chosen-version-cant-derive)={len(skipped)}",
          flush=True)

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
    for t, full, gen, out in zip(targets, fulls, gens, outs):
        ptoks = out.prompt_token_ids
        plps = out.prompt_logprobs
        comp_start = len(tok(gen, add_special_tokens=False)["input_ids"])
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc["offset_mapping"]
        bounds = region_boundaries(full, len(gen), markers)
        tokens, first_div = [], None
        for pos in range(comp_start, len(ptoks)):
            tid = ptoks[pos]
            entry = plps[pos] if pos < len(plps) else None
            if not entry or tid not in entry:
                continue
            rank = entry[tid].rank
            char_off = offsets[pos][0] if pos < len(offsets) else len(gen)
            reg = region_at(char_off, bounds)
            d = {"pos": pos - comp_start, "region": reg,
                 "gold": tok.decode([tid]), "logprob": round(float(entry[tid].logprob), 3),
                 "rank": rank}
            if rank != 1:
                d["argmax"] = min(entry.values(), key=lambda x: x.rank).decoded_token
            tokens.append(d)
            if first_div is None and rank != 1:
                first_div = d
        results.append({
            "id": t["id"], "baseline_failing": t["baseline_failing"],
            "n_comp_tokens": len(tokens),
            "n_mismatch": sum(1 for x in tokens if x["rank"] != 1),
            "clean": all(x["rank"] == 1 for x in tokens),
            "first_divergence": first_div,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"adapter": args.adapter, "trace_version": args.trace_version,
         "rule_region": rule_region, "skipped": skipped, "results": results}, indent=2))

    failing = [r for r in results if r["baseline_failing"]]
    pool = failing if args.restrict_failing else results
    hist = collections.Counter(
        (r["first_divergence"]["region"] if r["first_divergence"] else "NONE(clean)")
        for r in pool)
    print(f"\n[probe] wrote {args.out}")
    print(f"[probe] probed {len(pool)} ({'previously-failing' if args.restrict_failing else 'all covered'})")
    print(f"[probe] clean (no divergence): {sum(1 for r in pool if r['clean'])}/{len(pool)}")
    print(f"[probe] first-divergence in rule region '{rule_region}': "
          f"{hist.get(rule_region, 0)}/{len(pool)}")
    print("[probe] first-divergence region histogram:")
    for reg, c in hist.most_common():
        print(f"    {c:3d}  {reg}")
    if skipped:
        print(f"[probe] note: {len(skipped)} previously-failing IDs are not "
              f"derivable by {args.trace_version} (excluded): {skipped[:8]}")


if __name__ == "__main__":
    main()
