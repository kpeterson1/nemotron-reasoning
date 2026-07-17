"""Expanded bit_manipulation solver (v2).

Two-stage strategy:

  (A) **Shift-invariant search** — the existing K∈{1,2,3} solver from v1.
      output[i] = f(input[i+d_1], ..., input[i+d_K]) for all i, with either
      wrap-around or zero-padded edges. This handles ~70% of puzzles with
      ~98% generalization to the held-out test input.

  (B) **Per-output-bit decomposition** (new) — if no global invariant fits,
      decompose into 8 independent classification problems (one per output
      bit position) and find a small rule for each. Candidate rules:

         K=0 constant:                      2 (always 0, always 1)
         K=1 copy/not from input position:  16 (8 positions × {x, ¬x})
         K=2 natural Boolean fn:            280 (28 pairs × 10 non-degenerate
                                            2-input truth tables)
         K=3 natural Boolean fn:             ~280 (56 triples × 5 functions:
                                            MAJ, OR3, AND3, XOR3, Ch)
       Total: ~578 candidates per output bit.

      For each output position, find the simplest rule (lowest K) that fits
      every example. If multiple K=2 rules fit, prefer the one that appears
      most often across the 8 output positions (position-uniformity heuristic
      that mirrors shift-invariance even when the rule isn't strictly a
      shift).

  (C) If even per-bit can't fit all 8 positions, abandon (return None).

The downstream training driver verifies each solver answer against the GT
before keeping the example, so over-fitting per-bit rules are filtered out
naturally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


# ---- public API ----

@dataclass(frozen=True)
class InvariantRule:
    kind: str = "invariant"
    offsets: tuple[int, ...] = ()
    truth_table: int = 0
    wrap: str = "zero"
    k: int = 0


@dataclass(frozen=True)
class PerBitRule:
    """Per-output-bit rule. `bits[p]` is a tuple describing the rule for
    output position p:
      ('const', value)
      ('copy', input_pos)
      ('not', input_pos)
      ('k2', (i, j), truth_table)
      ('k3', (i, j, k), truth_table)
    """
    kind: str = "per_bit"
    bits: tuple = ()


def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    m = re.search(r"Now,\s+determine\s+the\s+output\s+for:\s*([01]{8})", prompt)
    return pairs, (m.group(1) if m else None)


# ---- shift-invariant search (same as v1; included so this module is
# standalone) -------------------------------------------------------------

def _shift(arr: np.ndarray, d: int, wrap: str) -> np.ndarray:
    if d == 0:
        return arr.copy()
    if wrap == "wrap":
        if d > 0:
            return np.uint8((arr.astype(np.uint16) << d | arr.astype(np.uint16) >> (8 - d)) & 0xFF)
        d_ = -d
        return np.uint8((arr.astype(np.uint16) >> d_ | arr.astype(np.uint16) << (8 - d_)) & 0xFF)
    if d > 0:
        return np.uint8((arr.astype(np.uint16) << d) & 0xFF)
    return np.uint8(arr >> (-d))


def _is_degenerate(tt: int, k: int) -> bool:
    n = 1 << k
    for j in range(k):
        mask = 1 << j
        if all(((tt >> i) & 1) == ((tt >> (i ^ mask)) & 1) for i in range(n)):
            return True
    return False


def _eval_2(a, b, tt):
    masks = [(~a) & (~b) & 0xFF, a & (~b) & 0xFF, (~a) & b & 0xFF, a & b & 0xFF]
    pred = np.zeros_like(a)
    for m in range(4):
        if (tt >> m) & 1:
            pred = pred | masks[m]
    return pred


def _eval_3(a, b, c, tt):
    masks = []
    for m in range(8):
        am = a if m & 1 else (~a) & 0xFF
        bm = b if m & 2 else (~b) & 0xFF
        cm = c if m & 4 else (~c) & 0xFF
        masks.append(am & bm & cm & 0xFF)
    pred = np.zeros_like(a)
    for m in range(8):
        if (tt >> m) & 1:
            pred = pred | masks[m]
    return pred


def _find_tt_2(a, b, target):
    for tt in range(16):
        if _is_degenerate(tt, 2):
            continue
        if np.array_equal(_eval_2(a, b, tt), target):
            return tt
    return None


def _find_tt_3(a, b, c, target):
    for tt in range(256):
        if _is_degenerate(tt, 3):
            continue
        if np.array_equal(_eval_3(a, b, c, tt), target):
            return tt
    return None


def _search_invariant(pairs: list[tuple[str, str]]) -> InvariantRule | None:
    inputs = np.array([int(i, 2) for i, _ in pairs], dtype=np.uint8)
    outputs = np.array([int(o, 2) for _, o in pairs], dtype=np.uint8)
    for wrap in ("wrap", "zero"):
        for d in range(-7, 8):
            a = _shift(inputs, d, wrap)
            if np.array_equal(a, outputs):
                return InvariantRule(offsets=(d,), truth_table=0b10, wrap=wrap, k=1)
            if np.array_equal((~a) & 0xFF, outputs):
                return InvariantRule(offsets=(d,), truth_table=0b01, wrap=wrap, k=1)
        for d0 in range(-7, 8):
            a = _shift(inputs, d0, wrap)
            for d1 in range(d0 + 1, 8):
                b = _shift(inputs, d1, wrap)
                tt = _find_tt_2(a, b, outputs)
                if tt is not None:
                    return InvariantRule(offsets=(d0, d1), truth_table=tt, wrap=wrap, k=2)
        for d0 in range(-7, 8):
            a = _shift(inputs, d0, wrap)
            for d1 in range(d0 + 1, 8):
                b = _shift(inputs, d1, wrap)
                for d2 in range(d1 + 1, 8):
                    c = _shift(inputs, d2, wrap)
                    tt = _find_tt_3(a, b, c, outputs)
                    if tt is not None:
                        return InvariantRule(
                            offsets=(d0, d1, d2), truth_table=tt, wrap=wrap, k=3
                        )
    return None


def _apply_invariant(rule: InvariantRule, x: int) -> int:
    arr = np.array([x], dtype=np.uint8)
    a = _shift(arr, rule.offsets[0], rule.wrap)
    if rule.k == 1:
        return int(a[0]) if rule.truth_table == 0b10 else int((~a[0]) & 0xFF)
    b = _shift(arr, rule.offsets[1], rule.wrap)
    if rule.k == 2:
        return int(_eval_2(a, b, rule.truth_table)[0])
    c = _shift(arr, rule.offsets[2], rule.wrap)
    return int(_eval_3(a, b, c, rule.truth_table)[0])


# ---- Per-bit decomposition ------------------------------------------------

# Natural 2-input truth tables (LSB-first input encoding); skip degenerate.
_NAT_K2 = [
    0b1000,  # AND
    0b1110,  # OR
    0b0110,  # XOR
    0b0111,  # NAND
    0b0001,  # NOR
    0b1001,  # XNOR
    0b0010,  # A AND NOT B
    0b0100,  # NOT A AND B
    0b1011,  # A OR NOT B
    0b1101,  # NOT A OR B
]
# Natural 3-input truth tables
def _k3_table(fn):
    return sum(int(fn(i & 1, (i >> 1) & 1, (i >> 2) & 1)) << i for i in range(8))


_NAT_K3 = [
    _k3_table(lambda a, b, c: a + b + c >= 2),           # MAJ
    _k3_table(lambda a, b, c: a or b or c),              # OR3
    _k3_table(lambda a, b, c: a & b & c),                # AND3
    _k3_table(lambda a, b, c: a ^ b ^ c),                # XOR3
    _k3_table(lambda a, b, c: (a & b) | ((1 - a) & c)),  # CH (SHA-style)
]


def _bits_msb_first(x: int) -> list[int]:
    return [(x >> (7 - i)) & 1 for i in range(8)]


def _eval_rule_one_bit(rule: tuple, in_bits: list[int]) -> int:
    tag = rule[0]
    if tag == "const":
        return rule[1]
    if tag == "copy":
        return in_bits[rule[1]]
    if tag == "not":
        return 1 - in_bits[rule[1]]
    if tag == "k2":
        _, (i, j), tt = rule
        idx = in_bits[i] | (in_bits[j] << 1)
        return (tt >> idx) & 1
    if tag == "k3":
        _, (i, j, k), tt = rule
        idx = in_bits[i] | (in_bits[j] << 1) | (in_bits[k] << 2)
        return (tt >> idx) & 1
    raise ValueError(f"unknown rule {rule!r}")


def _rule_complexity(rule: tuple) -> int:
    """Lower = simpler. Used to pick the simplest fitting rule per bit."""
    tag = rule[0]
    return {"const": 0, "copy": 1, "not": 1, "k2": 2, "k3": 3}[tag]


def _find_rule_for_bit(
    in_bits_per_ex: list[list[int]],
    target_per_ex: list[int],
) -> list[tuple]:
    """Return ALL rules that fit the given target across examples, in
    complexity-then-lexicographic order."""
    cands: list[tuple] = []
    # K=0 constants
    if all(t == 0 for t in target_per_ex):
        cands.append(("const", 0))
    if all(t == 1 for t in target_per_ex):
        cands.append(("const", 1))
    # K=1 copy / not from each input position
    for q in range(8):
        col = [b[q] for b in in_bits_per_ex]
        if col == target_per_ex:
            cands.append(("copy", q))
        if [1 - c for c in col] == target_per_ex:
            cands.append(("not", q))
    # K=2: 28 pairs × 10 natural truth tables
    for i, j in combinations(range(8), 2):
        ci = [b[i] for b in in_bits_per_ex]
        cj = [b[j] for b in in_bits_per_ex]
        for tt in _NAT_K2:
            ok = True
            for ex, target in enumerate(target_per_ex):
                idx = ci[ex] | (cj[ex] << 1)
                if ((tt >> idx) & 1) != target:
                    ok = False
                    break
            if ok:
                cands.append(("k2", (i, j), tt))
    # K=3: 56 triples × 5 natural truth tables
    for i, j, k in combinations(range(8), 3):
        ci = [b[i] for b in in_bits_per_ex]
        cj = [b[j] for b in in_bits_per_ex]
        ck = [b[k] for b in in_bits_per_ex]
        for tt in _NAT_K3:
            ok = True
            for ex, target in enumerate(target_per_ex):
                idx = ci[ex] | (cj[ex] << 1) | (ck[ex] << 2)
                if ((tt >> idx) & 1) != target:
                    ok = False
                    break
            if ok:
                cands.append(("k3", (i, j, k), tt))
    return cands


def _search_per_bit(pairs: list[tuple[str, str]]) -> PerBitRule | None:
    in_bits_per_ex = [_bits_msb_first(int(inp, 2)) for inp, _ in pairs]
    out_bits_per_ex = [_bits_msb_first(int(out, 2)) for _, out in pairs]

    # Step 1: gather candidate rule sets per output position
    per_bit_cands: list[list[tuple]] = []
    for p in range(8):
        target = [obs[p] for obs in out_bits_per_ex]
        cands = _find_rule_for_bit(in_bits_per_ex, target)
        if not cands:
            return None  # no rule fits this bit — abandon
        per_bit_cands.append(cands)

    # Step 2: score candidate rules by how many output positions they fit
    # across the whole puzzle (position-uniformity preference).
    rule_to_positions: dict[tuple, list[int]] = {}
    for p, cands in enumerate(per_bit_cands):
        for c in cands:
            rule_to_positions.setdefault(c, []).append(p)
    popularity = {c: len(positions) for c, positions in rule_to_positions.items()}

    # Step 3: for each position, pick the rule that is (a) simplest, then
    # (b) most popular, then (c) lexicographic.
    chosen: list[tuple] = []
    for p, cands in enumerate(per_bit_cands):
        cands_sorted = sorted(
            cands,
            key=lambda c: (_rule_complexity(c), -popularity[c], repr(c)),
        )
        chosen.append(cands_sorted[0])
    return PerBitRule(bits=tuple(chosen))


def _apply_per_bit(rule: PerBitRule, x: int) -> int:
    bits = _bits_msb_first(x)
    out = 0
    for p, sub in enumerate(rule.bits):
        v = _eval_rule_one_bit(sub, bits)
        out |= v << (7 - p)
    return out & 0xFF


# ---- Public solve() -------------------------------------------------------

def solve(pairs: Iterable[tuple[str, str]]):
    """Return an InvariantRule or PerBitRule, or None if no fit."""
    pairs = list(pairs)
    if not pairs:
        return None
    inv = _search_invariant(pairs)
    if inv is not None:
        return inv
    return _search_per_bit(pairs)


def apply_rule(rule, x: int) -> int:
    if isinstance(rule, InvariantRule):
        return _apply_invariant(rule, x)
    if isinstance(rule, PerBitRule):
        return _apply_per_bit(rule, x)
    raise TypeError(f"unknown rule type {type(rule).__name__}")


def describe_rule(rule) -> str:
    if isinstance(rule, InvariantRule):
        return f"invariant K={rule.k} offsets={rule.offsets} tt=0x{rule.truth_table:x} wrap={rule.wrap}"
    if isinstance(rule, PerBitRule):
        return "per_bit  " + " | ".join(repr(b) for b in rule.bits)
    return repr(rule)
