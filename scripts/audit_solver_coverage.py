"""Phase-1 audit: solver coverage + trace token budget on dev_frozen.

Read-only. Replicates the v9 build verification path (parse -> solve -> trace,
keep iff trace's answer == ground truth) for bit_manipulation and
equation_transformation against datasets/splits/dev_frozen.jsonl.

For each problem reports: trace produced? answer correct? trace token count
(flagging >6000, the inference-truncation danger zone). Classifies covered
problems by solver rule subtype and misses by failure mode / problem subtype.

Usage:
    python -m scripts.audit_solver_coverage            # both categories
    python -m scripts.audit_solver_coverage --json out.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.data.bit_manip_solver_v3 import (
    InvariantRule,
    PerBitRule,
    parse_examples as bm_parse,
    solve as bm_solve,
)
from src.data.bit_manip_trace_v4 import generate_trace as bm_trace
from src.data.equation_transformation_solver_v2 import (
    parse as eq_parse,
    solve as eq_solve,
    _detect_test_operator,
    _split_by_op,
)
from src.data.equation_transformation_trace_v2 import generate_trace as eq_trace
from src.data.text_encryption_solver import parse as te_parse
from src.data.text_encryption_trace import generate_trace as te_trace

DEV = Path("datasets/splits/dev_frozen.jsonl")
TOKEN_DANGER = 6000

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
        )
    return _TOK


def _n_tokens(text: str) -> int:
    return len(_tok().encode(text))


def load(task: str) -> list[dict]:
    rows = []
    with DEV.open() as f:
        for line in f:
            r = json.loads(line)
            if r["task_type"] == task:
                rows.append(r)
    return rows


# ---- bit_manipulation -----------------------------------------------------

def _bm_subtype(rule) -> str:
    if isinstance(rule, InvariantRule):
        return f"invariant_k{rule.k}_{rule.wrap}"
    if isinstance(rule, PerBitRule):
        fams = {a.family for a in rule.atoms}
        return "per_bit(" + ",".join(sorted(fams)) + ")"
    return "unknown_rule"


def audit_bit_manip() -> dict:
    rows = load("bit_manipulation")
    recs = []
    for r in rows:
        gt = str(r["answer"]).strip()
        pairs, q = bm_parse(r["prompt"])
        rec = {
            "id": r["id"], "n_examples": len(pairs), "gt": gt,
            "trace": False, "correct": False, "tokens": None,
            "subtype": None, "miss_reason": None,
        }
        if not pairs or q is None:
            rec["miss_reason"] = "parse_fail"
            recs.append(rec); continue
        rule = bm_solve(pairs)
        if rule is None:
            rec["miss_reason"] = "no_rule_found"
            recs.append(rec); continue
        res = bm_trace(rule, pairs, q)
        if res is None:
            rec["miss_reason"] = "trace_fail"
            recs.append(rec); continue
        trace, pred = res
        rec["trace"] = True
        rec["tokens"] = _n_tokens(trace)
        rec["subtype"] = _bm_subtype(rule)
        if pred == gt:
            rec["correct"] = True
        else:
            rec["miss_reason"] = "wrong_answer"
        recs.append(rec)
    return {"task": "bit_manipulation", "records": recs}


# ---- equation_transformation ----------------------------------------------

def _eq_subtype(prompt: str, ok: bool) -> str:
    test = eq_parse(prompt)
    if test is None:
        return "no_test_parsed"
    op = _detect_test_operator(test)
    if op is None:
        return "symbolic_no_op"
    parts = _split_by_op(test, op)
    if parts is None:
        return f"op[{op}]_unsplit"
    a, b = parts
    if a.isdigit() and b.isdigit():
        return f"numeric_op[{op}]"
    return f"nonnumeric_operands[{op}]"


def audit_eq() -> dict:
    rows = load("equation_transformation")
    recs = []
    for r in rows:
        gt = str(r["answer"]).strip()
        rec = {
            "id": r["id"], "gt": gt, "trace": False, "correct": False,
            "tokens": None, "subtype": None, "op_name": None, "miss_reason": None,
        }
        # solver-level rule (for subtype on covered)
        try:
            ans, rule = eq_solve(r["prompt"])
        except Exception:
            ans, rule = None, None
        res = None
        try:
            res = eq_trace(r["prompt"])
        except Exception:
            res = None
        if res is None:
            rec["miss_reason"] = "no_trace"
            rec["subtype"] = _eq_subtype(r["prompt"], False)
            recs.append(rec); continue
        trace, answer = res
        rec["trace"] = True
        rec["tokens"] = _n_tokens(trace)
        rec["op_name"] = rule.op_name if rule is not None else None
        if str(answer).strip() == gt:
            rec["correct"] = True
            rec["subtype"] = (rule.op_name if rule is not None else "symbolic") if rule else "symbolic"
        else:
            rec["miss_reason"] = "wrong_answer"
            rec["subtype"] = _eq_subtype(r["prompt"], False)
        recs.append(rec)
    return {"task": "equation_transformation", "records": recs}



# ---- text_encryption (v9 original trace format) ---------------------------

def audit_text_enc() -> dict:
    """Coverage of the v9 text_encryption trace (the shipped format), same
    keep-iff-answer-matches rule as the v9 build path. CPU-only."""
    rows = load("text_encryption")
    recs = []
    for r in rows:
        gt = str(r["answer"]).strip()
        rec = {
            "id": r["id"], "gt": gt, "trace": False, "correct": False,
            "tokens": None, "subtype": None, "miss_reason": None,
        }
        try:
            pairs, _test = te_parse(r["prompt"])
        except Exception:
            pairs = []
        rec["subtype"] = f"examples_{len(pairs)}"
        try:
            res = te_trace(r["prompt"])
        except Exception:
            res = None
        if res is None:
            rec["miss_reason"] = "no_trace"
            recs.append(rec); continue
        trace, answer = res
        rec["trace"] = True
        rec["tokens"] = _n_tokens(trace)
        if str(answer).strip() == gt:
            rec["correct"] = True
        else:
            rec["miss_reason"] = "wrong_answer"
        recs.append(rec)
    return {"task": "text_encryption", "records": recs}


# ---- reporting ------------------------------------------------------------

def report(audit: dict) -> None:
    recs = audit["records"]
    n = len(recs)
    correct = [r for r in recs if r["correct"]]
    no_trace = [r for r in recs if not r["trace"]]
    wrong = [r for r in recs if r["trace"] and not r["correct"]]
    over = [r for r in correct if r["tokens"] and r["tokens"] > TOKEN_DANGER]
    print(f"\n===== {audit['task']} =====")
    print(f"total problems            : {n}")
    print(f"covered (correct trace)   : {len(correct):3d}  ({100*len(correct)/n:.1f}%)")
    print(f"  of which >6000 tokens   : {len(over):3d}  ({100*len(over)/n:.1f}%)")
    print(f"miss: no trace            : {len(no_trace):3d}  ({100*len(no_trace)/n:.1f}%)")
    print(f"miss: wrong answer        : {len(wrong):3d}  ({100*len(wrong)/n:.1f}%)")

    if correct:
        toks = sorted(r["tokens"] for r in correct)
        print(f"covered trace tokens: min={toks[0]} med={toks[len(toks)//2]} "
              f"max={toks[-1]}")
    print("-- covered by subtype --")
    for st, c in collections.Counter(r["subtype"] for r in correct).most_common():
        over_c = sum(1 for r in correct if r["subtype"] == st and r["tokens"] and r["tokens"] > TOKEN_DANGER)
        print(f"  {c:3d}  {st}   (>6k: {over_c})")
    print("-- miss: no trace, by subtype --")
    for st, c in collections.Counter(r["subtype"] for r in no_trace).most_common():
        print(f"  {c:3d}  {st}")
    print("-- miss: wrong answer, by subtype --")
    for st, c in collections.Counter(r["subtype"] for r in wrong).most_common():
        print(f"  {c:3d}  {st}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--tasks", nargs="+",
        choices=["bit_manipulation", "equation_transformation", "text_encryption"],
        default=["bit_manipulation", "equation_transformation"],
        help="which category audits to run (default preserves original behavior)",
    )
    args = ap.parse_args()
    fns = {
        "bit_manipulation": audit_bit_manip,
        "equation_transformation": audit_eq,
        "text_encryption": audit_text_enc,
    }
    audits = [fns[t]() for t in args.tasks]
    for a in audits:
        report(a)
    if args.json:
        args.json.write_text(json.dumps(audits, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
