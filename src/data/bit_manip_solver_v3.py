"""Per-output-bit decomposition solver for `bit_manipulation` puzzles.

Independent implementation of the per-output-bit strategy. Each output bit
position p (0..7, MSB-first) is treated as an independent classification
problem: pick a Boolean rule taking input bits at chosen positions to the
observed target column.

Two stages:

  A. Shift-invariant search (K=1,2,3 with wrap- or zero-pad). Fast and exact
     for puzzles whose rule applies uniformly to every output bit.

  B. Per-output-bit decomposition. For each position p, collect candidate
     rules in 9 families:
        Identity  : copy input bit q
        NOT       : invert input bit q
        Constant  : 0 or 1
        AND, OR, XOR              (symmetric K=2 over 28 unordered pairs)
        AND-NOT, OR-NOT, XOR-NOT  (asymmetric K=2 over 56 ordered pairs;
                                  the second operand is inverted first)

     The selection prefers a single family that covers all 8 output
     positions ("homogeneous"), then falls back to mixing families with
     the longest stride-run as the anchor. K=3 operators are intentionally
     excluded — they over-fit small example sets, and the dataset's puzzles
     are dominantly K<=2.

Public API:
    pairs, question_bits = parse_examples(prompt)
    rule = solve(pairs)
    out_int = apply_rule(rule, in_int)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np

N_BITS = 8

# Truth-table indices use MSB-first encoding of the operand bits, i.e.
# for K=2 with operands (a, b), the truth-table bit at position (a*2 + b)
# gives the output. NOT-flavoured families invert the *second* operand
# before evaluation.

# Maps family -> base truth table (symmetric ops are evaluated on (a,b);
# *-NOT ops are evaluated on (a, ~b)).
_FAMILY_TT: dict[str, int] = {
    # bit pattern for (a,b) read MSB-first: 00 01 10 11
    "AND":   0b0001,  # a & b
    "OR":    0b0111,  # a | b
    "XOR":   0b0110,  # a ^ b
    # *-NOT families: input is (a, ~b). Same boolean function name applies.
    "AND-NOT": 0b0001,
    "OR-NOT":  0b0111,
    "XOR-NOT": 0b0110,
}

_SYM_FAMILIES = ("AND", "OR", "XOR", "NAND", "NOR")
_ASYM_FAMILIES = ("AND-NOT", "OR-NOT", "XOR-NOT")
_PAIR_FAMILIES = _SYM_FAMILIES + _ASYM_FAMILIES

# Named K=3 boolean functions. Used as a fallback per-bit family when no
# K<=2 atom fits a column. Bit `m` of the TT corresponds to operand vector
# (a, b, c) = (m&1, (m>>1)&1, (m>>2)&1).
_TRIPLE_FAMILIES = ("MAJ", "OR3", "AND3", "XOR3", "CH")
_TRIPLE_TT: dict[str, int] = {
    "MAJ":  sum(1 << m for m in range(8) if bin(m).count("1") >= 2),
    "OR3":  sum(1 << m for m in range(8) if m != 0),
    "AND3": 1 << 7,
    "XOR3": sum(1 << m for m in range(8) if bin(m).count("1") % 2 == 1),
    # CH (SHA-style): c if a else b → (a & b) | (~a & c). Using m = a + 2b + 4c
    # this is bit 2 (c=0,b=0,a=0 → 0), 0,1,1,0,1,1,1 etc.
    "CH":   sum(
        1 << m for m in range(8)
        if (((m & 1) and ((m >> 1) & 1)) or ((not (m & 1)) and ((m >> 2) & 1)))
    ),
}

# Global families: rule depends on ALL 8 input bits (parity, popcount-threshold).
# Operand index is a tag identifying the specific global function.
_GLOBAL_FAMILIES = ("Parity", "Popcount")

# Section order for homogeneous-family selection. Simpler families first.
_SECTIONS = (
    ("Identity", "NOT", "Constant") + _PAIR_FAMILIES + _TRIPLE_FAMILIES + _GLOBAL_FAMILIES
)


# ---- Rule types -----------------------------------------------------------

@dataclass(frozen=True)
class InvariantRule:
    kind: str = "invariant"
    offsets: tuple[int, ...] = ()
    truth_table: int = 0
    wrap: str = "zero"
    k: int = 0


@dataclass(frozen=True)
class PerBitAtom:
    """One per-position rule. `family` plus a tuple of input-bit positions."""
    family: str           # Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT,
                          # MAJ, OR3, AND3, XOR3, CH
    operands: tuple       # Identity/NOT: (q,); Constant: ("0",)/("1",);
                          # pair: (i, j); triple: (i, j, k)


@dataclass(frozen=True)
class PerBitRule:
    kind: str = "per_bit"
    atoms: tuple = ()     # 8 PerBitAtom


# ---- Prompt parsing -------------------------------------------------------

def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    m = re.search(r"determine\s+the\s+output\s+for:\s*([01]{8})", prompt)
    return pairs, (m.group(1) if m else None)


# ---- Stage A: shift-invariant search --------------------------------------

def _shift_arr(arr: np.ndarray, d: int, wrap: str) -> np.ndarray:
    """Right-rotate-or-shift each byte by `d` MSB-positions. y[i] = x[i+d]
    in MSB-first indexing, edge-handled per `wrap`."""
    if d == 0:
        return arr.copy()
    if wrap == "wrap":
        if d > 0:
            return np.uint8(
                ((arr.astype(np.uint16) << d) | (arr.astype(np.uint16) >> (N_BITS - d)))
                & 0xFF
            )
        d_ = -d
        return np.uint8(
            ((arr.astype(np.uint16) >> d_) | (arr.astype(np.uint16) << (N_BITS - d_)))
            & 0xFF
        )
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


def _eval_pair_tt(a: np.ndarray, b: np.ndarray, tt: int) -> np.ndarray:
    masks = [
        (~a) & (~b) & 0xFF,
        a & (~b) & 0xFF,
        (~a) & b & 0xFF,
        a & b & 0xFF,
    ]
    out = np.zeros_like(a)
    for m in range(4):
        if (tt >> m) & 1:
            out = out | masks[m]
    return out


def _eval_triple_tt(a: np.ndarray, b: np.ndarray, c: np.ndarray, tt: int) -> np.ndarray:
    out = np.zeros_like(a)
    for m in range(8):
        am = a if m & 1 else (~a) & 0xFF
        bm = b if m & 2 else (~b) & 0xFF
        cm = c if m & 4 else (~c) & 0xFF
        if (tt >> m) & 1:
            out = out | (am & bm & cm & 0xFF)
    return out


def _infer_tt(operand_bits: np.ndarray, target_bits: np.ndarray, k: int) -> int | None:
    """Given operand bit columns and target bit column (each shape (M,) of
    0/1 ints), derive the unique truth table over k inputs that maps every
    minterm to a consistent target value. Returns None if inconsistent or
    if some minterm is unconstrained (no example reaches it) — in that case
    we'd have a non-unique fit, which we reject.

    Returns the TT as an int, or None.
    """
    n_minterms = 1 << k
    # Build minterm index for each (example, bit position) sample.
    # operand_bits has shape (k, M) with values in {0,1}.
    weights = (1 << np.arange(k)).reshape(k, 1)
    minterms = (operand_bits * weights).sum(axis=0)
    # For each minterm value, all target_bits must agree.
    tt = 0
    seen = [False] * n_minterms
    out_for_minterm = [0] * n_minterms
    for s in range(minterms.shape[0]):
        m = int(minterms[s])
        t = int(target_bits[s])
        if seen[m]:
            if out_for_minterm[m] != t:
                return None
        else:
            seen[m] = True
            out_for_minterm[m] = t
    for m in range(n_minterms):
        if not seen[m]:
            return None  # under-constrained — reject
        if out_for_minterm[m]:
            tt |= (1 << m)
    return tt


def _unpack_bits(byte_arr: np.ndarray) -> np.ndarray:
    """(M,) uint8 -> (M, 8) array of 0/1 ints, MSB-first."""
    return np.unpackbits(byte_arr.reshape(-1, 1), axis=1).astype(np.int8)


def _search_invariant(pairs: list[tuple[str, str]]) -> InvariantRule | None:
    ins = np.array([int(i, 2) for i, _ in pairs], dtype=np.uint8)
    outs = np.array([int(o, 2) for _, o in pairs], dtype=np.uint8)

    # Precompute per-bit unpacked arrays for every shift × wrap.
    shifted_bits: dict[tuple[int, str], np.ndarray] = {}
    for wrap in ("wrap", "zero"):
        for d in range(-7, 8):
            shifted_bits[(d, wrap)] = _unpack_bits(_shift_arr(ins, d, wrap)).flatten()
    out_flat = _unpack_bits(outs).flatten()

    for wrap in ("wrap", "zero"):
        # K=1 — keep fast equality check on packed bytes.
        for d in range(-7, 8):
            a = _shift_arr(ins, d, wrap)
            if np.array_equal(a, outs):
                return InvariantRule(offsets=(d,), truth_table=0b10, wrap=wrap, k=1)
            if np.array_equal((~a) & 0xFF, outs):
                return InvariantRule(offsets=(d,), truth_table=0b01, wrap=wrap, k=1)
        # K=2 — infer truth table directly.
        for d0 in range(-7, 8):
            a_bits = shifted_bits[(d0, wrap)]
            for d1 in range(d0 + 1, 8):
                b_bits = shifted_bits[(d1, wrap)]
                tt = _infer_tt(np.stack([a_bits, b_bits]), out_flat, 2)
                if tt is not None and not _is_degenerate(tt, 2):
                    return InvariantRule(
                        offsets=(d0, d1), truth_table=tt, wrap=wrap, k=2
                    )
        # K=3 — infer truth table directly. Skip if too few examples to cover
        # 8 minterms (TT would be under-constrained).
        n_samples = out_flat.shape[0]
        if n_samples < 8:
            continue
        for d0 in range(-7, 8):
            a_bits = shifted_bits[(d0, wrap)]
            for d1 in range(d0 + 1, 8):
                b_bits = shifted_bits[(d1, wrap)]
                for d2 in range(d1 + 1, 8):
                    c_bits = shifted_bits[(d2, wrap)]
                    tt = _infer_tt(
                        np.stack([a_bits, b_bits, c_bits]), out_flat, 3
                    )
                    if tt is not None and not _is_degenerate(tt, 3):
                        return InvariantRule(
                            offsets=(d0, d1, d2),
                            truth_table=tt,
                            wrap=wrap,
                            k=3,
                        )
    return None


def _apply_invariant(rule: InvariantRule, x: int) -> int:
    arr = np.array([x], dtype=np.uint8)
    a = _shift_arr(arr, rule.offsets[0], rule.wrap)
    if rule.k == 1:
        return int(a[0]) if rule.truth_table == 0b10 else int((~a[0]) & 0xFF)
    b = _shift_arr(arr, rule.offsets[1], rule.wrap)
    if rule.k == 2:
        return int(_eval_pair_tt(a, b, rule.truth_table)[0])
    c = _shift_arr(arr, rule.offsets[2], rule.wrap)
    return int(_eval_triple_tt(a, b, c, rule.truth_table)[0])


# ---- Stage B: per-output-bit decomposition --------------------------------

def _bits_msb(x: int) -> tuple[int, ...]:
    return tuple((x >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS))


def _evaluate_atom(atom: PerBitAtom, in_bits: tuple[int, ...]) -> int:
    fam = atom.family
    if fam == "Constant":
        return 1 if atom.operands[0] == "1" else 0
    if fam == "Identity":
        return in_bits[atom.operands[0]]
    if fam == "NOT":
        return 1 - in_bits[atom.operands[0]]
    if fam == "Parity":
        # operands = (variant,) where 0 = parity, 1 = not parity.
        bit = sum(in_bits) & 1
        return bit if atom.operands[0] == 0 else 1 - bit
    if fam == "Popcount":
        # operands = (op, threshold) where op is "ge", "le", or "eq".
        op, thr = atom.operands
        n = sum(in_bits)
        if op == "ge":
            return 1 if n >= thr else 0
        if op == "le":
            return 1 if n <= thr else 0
        return 1 if n == thr else 0
    if fam in _TRIPLE_FAMILIES:
        i, j, k = atom.operands
        a, b, c = in_bits[i], in_bits[j], in_bits[k]
        m = a | (b << 1) | (c << 2)
        return (_TRIPLE_TT[fam] >> m) & 1
    i, j = atom.operands
    a = in_bits[i]
    b = in_bits[j]
    if fam in _ASYM_FAMILIES:
        b = 1 - b
    if fam == "NAND":
        return 1 - (a & b)
    if fam == "NOR":
        return 1 - (a | b)
    base_fam = fam.split("-")[0]
    if base_fam == "AND":
        return a & b
    if base_fam == "OR":
        return a | b
    if base_fam == "XOR":
        return a ^ b
    raise ValueError(f"unknown family {fam!r}")


def _gather_atoms_for_bit(
    in_cols: list[tuple[int, ...]],
    out_col: tuple[int, ...],
) -> dict[str, list[PerBitAtom]]:
    """Return, per family, every PerBitAtom that reproduces `out_col`.

    `in_cols[q]` is the tuple of input bits at position q across all
    examples; `out_col` is the same for the output column being matched.
    """
    n_ex = len(out_col)
    candidates: dict[str, list[PerBitAtom]] = {fam: [] for fam in _SECTIONS}

    # Constants
    if all(v == 0 for v in out_col):
        candidates["Constant"].append(PerBitAtom("Constant", ("0",)))
    if all(v == 1 for v in out_col):
        candidates["Constant"].append(PerBitAtom("Constant", ("1",)))

    # Identity / NOT
    for q in range(N_BITS):
        col_q = in_cols[q]
        if col_q == out_col:
            candidates["Identity"].append(PerBitAtom("Identity", (q,)))
        if tuple(1 - b for b in col_q) == out_col:
            candidates["NOT"].append(PerBitAtom("NOT", (q,)))

    # Symmetric pair families: try all 28 unordered (i, j) pairs.
    for i, j in combinations(range(N_BITS), 2):
        ai = in_cols[i]
        aj = in_cols[j]
        for fam in _SYM_FAMILIES:
            ok = True
            for k in range(n_ex):
                a, b = ai[k], aj[k]
                if fam == "AND":
                    pred = a & b
                elif fam == "OR":
                    pred = a | b
                elif fam == "XOR":
                    pred = a ^ b
                elif fam == "NAND":
                    pred = 1 - (a & b)
                elif fam == "NOR":
                    pred = 1 - (a | b)
                if pred != out_col[k]:
                    ok = False
                    break
            if ok:
                candidates[fam].append(PerBitAtom(fam, (i, j)))

    # Asymmetric pair families: try all 56 ordered (i, j) pairs.
    for i in range(N_BITS):
        for j in range(N_BITS):
            if i == j:
                continue
            ai = in_cols[i]
            aj_neg = tuple(1 - v for v in in_cols[j])
            for fam in _ASYM_FAMILIES:
                ok = True
                for k in range(n_ex):
                    a, b = ai[k], aj_neg[k]
                    if fam == "AND-NOT":
                        pred = a & b
                    elif fam == "OR-NOT":
                        pred = a | b
                    else:
                        pred = a ^ b
                    if pred != out_col[k]:
                        ok = False
                        break
                if ok:
                    candidates[fam].append(PerBitAtom(fam, (i, j)))

    # Global families: rule depends on the popcount or parity of ALL inputs.
    # These trigger for puzzles where the output column is constant per
    # popcount value (e.g. parity bit, "majority of input bits", etc.).
    popcounts = [sum(in_bits) for in_bits in zip(*in_cols)]
    parity_col = tuple(p & 1 for p in popcounts)
    if parity_col == out_col:
        candidates["Parity"].append(PerBitAtom("Parity", (0,)))
    if tuple(1 - b for b in parity_col) == out_col:
        candidates["Parity"].append(PerBitAtom("Parity", (1,)))
    for thr in range(1, N_BITS):
        ge_col = tuple(1 if p >= thr else 0 for p in popcounts)
        le_col = tuple(1 if p <= thr else 0 for p in popcounts)
        eq_col = tuple(1 if p == thr else 0 for p in popcounts)
        if ge_col == out_col:
            candidates["Popcount"].append(PerBitAtom("Popcount", ("ge", thr)))
        if le_col == out_col:
            candidates["Popcount"].append(PerBitAtom("Popcount", ("le", thr)))
        if eq_col == out_col:
            candidates["Popcount"].append(PerBitAtom("Popcount", ("eq", thr)))

    # K=3 named triple families. Enumerated over 56 unordered triples. Used
    # as a deeper fallback — the MSB-symmetry of MAJ/OR3/AND3/XOR3 means the
    # operand order doesn't affect the output. CH is asymmetric, so we'd need
    # all 336 ordered triples to fully cover it; for cost reasons we keep
    # the unordered enumeration and rely on the per-bit fallback for CH
    # cases that don't land on the chosen ordering.
    for i, j, k in combinations(range(N_BITS), 3):
        col_i = in_cols[i]
        col_j = in_cols[j]
        col_k = in_cols[k]
        for fam in _TRIPLE_FAMILIES:
            tt = _TRIPLE_TT[fam]
            ok = True
            for s in range(n_ex):
                m = col_i[s] | (col_j[s] << 1) | (col_k[s] << 2)
                if ((tt >> m) & 1) != out_col[s]:
                    ok = False
                    break
            if ok:
                candidates[fam].append(PerBitAtom(fam, (i, j, k)))

    return candidates


def _atom_complexity(atom: PerBitAtom) -> int:
    fam = atom.family
    if fam == "Constant":
        return 0
    if fam in ("Identity", "NOT"):
        return 1
    return 2


def _atom_repr_key(atom: PerBitAtom) -> tuple:
    """Stable tie-breaker between equivalently-complex atoms."""
    return (_atom_complexity(atom), atom.family, atom.operands)


def _longest_stride_run(
    atoms_by_pos: dict[int, list[PerBitAtom]],
    family: str,
) -> int:
    """Longest contiguous run of output positions p, p+1, ... where the atom
    at p+t has operands shifted by +t (mod 8) from the start.

    For Constant family, every position trivially matches (no operands), so
    we just return N_BITS if the family covers every position; otherwise 0.
    Used as a tie-breaker between families that fit all 8 positions.
    """
    if family == "Constant":
        return N_BITS if all(
            any(a.family == "Constant" for a in atoms_by_pos.get(p, []))
            for p in range(N_BITS)
        ) else 0

    best = 0
    for start in range(N_BITS):
        for atom in atoms_by_pos.get(start, []):
            if atom.family != family:
                continue
            length = 1
            p = atom.operands[0]
            q = atom.operands[1] if family in _PAIR_FAMILIES else None
            for t in range(1, N_BITS - start):
                ep = (p + t) % N_BITS
                eq = (q + t) % N_BITS if q is not None else None
                found = False
                for a2 in atoms_by_pos.get(start + t, []):
                    if a2.family != family:
                        continue
                    if a2.operands[0] != ep:
                        continue
                    if family in _PAIR_FAMILIES and a2.operands[1] != eq:
                        continue
                    found = True
                    break
                if not found:
                    break
                length += 1
            if length > best:
                best = length
    return best


def _stride_atom_at(
    atoms_at_pos: list[PerBitAtom],
    family: str,
    expected_p: int,
    expected_q: int | None,
) -> PerBitAtom | None:
    for a in atoms_at_pos:
        if a.family != family:
            continue
        if a.operands[0] != expected_p:
            continue
        if family in _PAIR_FAMILIES and a.operands[1] != expected_q:
            continue
        return a
    return None


def _left_run(
    atoms_per_bit: list[dict[str, list[PerBitAtom]]],
    family: str,
) -> list[PerBitAtom]:
    """Longest stride-+1 run starting at bit 0. Tries every starter atom at
    position 0 in `family` and returns the best chain."""
    if family == "Constant":
        # No operands to extrapolate; the "run" is just the prefix of positions
        # that have a Constant candidate.
        chain: list[PerBitAtom] = []
        for p in range(N_BITS):
            picks = [a for a in atoms_per_bit[p][family]]
            if not picks:
                break
            chain.append(picks[0])
        return chain

    if family in _GLOBAL_FAMILIES:
        # Match the same atom across consecutive positions starting at 0.
        chain: list[PerBitAtom] = []
        if not atoms_per_bit[0].get(family):
            return chain
        # Use the first atom at pos 0 and check if the SAME atom fits each
        # subsequent position.
        starter = atoms_per_bit[0][family][0]
        chain.append(starter)
        for p in range(1, N_BITS):
            if any(a == starter for a in atoms_per_bit[p].get(family, [])):
                chain.append(starter)
            else:
                break
        return chain

    best: list[PerBitAtom] = []
    for starter in atoms_per_bit[0].get(family, []):
        p0 = starter.operands[0]
        q0 = starter.operands[1] if family in _PAIR_FAMILIES else None
        chain = [starter]
        for t in range(1, N_BITS):
            ep = (p0 + t) % N_BITS
            eq = (q0 + t) % N_BITS if q0 is not None else None
            nxt = _stride_atom_at(atoms_per_bit[t].get(family, []), family, ep, eq)
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) > len(best):
            best = chain
    return best


def _right_run(
    atoms_per_bit: list[dict[str, list[PerBitAtom]]],
    family: str,
) -> list[PerBitAtom]:
    """Longest stride-+1 run ending at bit 7."""
    if family == "Constant":
        chain: list[PerBitAtom] = []
        for p in range(N_BITS - 1, -1, -1):
            picks = [a for a in atoms_per_bit[p][family]]
            if not picks:
                break
            chain.append(picks[0])
        return list(reversed(chain))

    if family in _GLOBAL_FAMILIES:
        chain: list[PerBitAtom] = []
        if not atoms_per_bit[N_BITS - 1].get(family):
            return chain
        ender = atoms_per_bit[N_BITS - 1][family][0]
        chain.append(ender)
        for p in range(N_BITS - 2, -1, -1):
            if any(a == ender for a in atoms_per_bit[p].get(family, [])):
                chain.insert(0, ender)
            else:
                break
        return chain

    best: list[PerBitAtom] = []
    for ender in atoms_per_bit[N_BITS - 1].get(family, []):
        p_last = ender.operands[0]
        q_last = ender.operands[1] if family in _PAIR_FAMILIES else None
        chain = [ender]
        for t in range(1, N_BITS):
            ep = (p_last - t) % N_BITS
            eq = (q_last - t) % N_BITS if q_last is not None else None
            nxt = _stride_atom_at(
                atoms_per_bit[N_BITS - 1 - t].get(family, []), family, ep, eq
            )
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) > len(best):
            best = chain
    return list(reversed(best))


def _search_per_bit(pairs: list[tuple[str, str]]) -> PerBitRule | None:
    in_bits_per_ex = [_bits_msb(int(inp, 2)) for inp, _ in pairs]
    out_bits_per_ex = [_bits_msb(int(out, 2)) for _, out in pairs]

    in_cols = [
        tuple(in_bits_per_ex[k][q] for k in range(len(pairs)))
        for q in range(N_BITS)
    ]
    out_cols = [
        tuple(out_bits_per_ex[k][p] for k in range(len(pairs)))
        for p in range(N_BITS)
    ]

    atoms_per_bit: list[dict[str, list[PerBitAtom]]] = []
    for p in range(N_BITS):
        per_fam = _gather_atoms_for_bit(in_cols, out_cols[p])
        if not any(per_fam.values()):
            return None
        atoms_per_bit.append(per_fam)

    # Step 1: compute best left/right runs across K<=2 families only —
    # K=3 atoms over-fit small example sets and produce poor question-
    # generalization, so they should never anchor a run.
    primary_sections = tuple(s for s in _SECTIONS if s not in _TRIPLE_FAMILIES)
    left_runs = {fam: _left_run(atoms_per_bit, fam) for fam in primary_sections}
    right_runs = {fam: _right_run(atoms_per_bit, fam) for fam in primary_sections}
    left_winner = max(
        primary_sections,
        key=lambda f: (len(left_runs[f]), -primary_sections.index(f)),
    )
    right_winner = max(
        primary_sections,
        key=lambda f: (len(right_runs[f]), -primary_sections.index(f)),
    )
    left_chain = left_runs[left_winner]
    right_chain = right_runs[right_winner]

    # Step 2: truncate so they don't overlap. Prefer the longer side.
    if len(left_chain) + len(right_chain) > N_BITS:
        if len(right_chain) >= len(left_chain):
            left_chain = left_chain[: N_BITS - len(right_chain)]
        else:
            right_chain = right_chain[len(right_chain) - (N_BITS - len(left_chain)):]

    # Step 3: place left and right runs into the 8-slot vector.
    chosen: list[PerBitAtom | None] = [None] * N_BITS
    for p, atom in enumerate(left_chain):
        chosen[p] = atom
    right_start = N_BITS - len(right_chain)
    for i, atom in enumerate(right_chain):
        chosen[right_start + i] = atom

    # Step 4: fill remaining (pending) positions.
    pending = [p for p in range(N_BITS) if chosen[p] is None]

    if pending:
        # Perfect match for the middle: first K<=2 family that covers every
        # pending position. K=3 families are excluded — they routinely over-
        # fit, so we use them only as a last resort below.
        chosen_middle_family: str | None = None
        for fam in primary_sections:
            if all(atoms_per_bit[p][fam] for p in pending):
                chosen_middle_family = fam
                break

        if chosen_middle_family is not None:
            # Stride hint: if the chosen middle family matches the family of
            # the longer side (left or right) and that side has at least one
            # atom, prefer middle atoms that continue the stride.
            stride_hint = _stride_hint(
                chosen_middle_family, left_chain, right_chain
            )
            for p in pending:
                cands = atoms_per_bit[p][chosen_middle_family]
                chosen[p] = _pick_stride_aware(cands, stride_hint, p)
        else:
            # Fall back per-position: simplest K<=2 atom; if none, allow K=3
            # for that one position only.
            for p in pending:
                picked: PerBitAtom | None = None
                for fam in primary_sections:
                    if atoms_per_bit[p][fam]:
                        picked = _pick_atom_from_family(atoms_per_bit[p][fam])
                        break
                if picked is None:
                    for fam in _TRIPLE_FAMILIES:
                        if atoms_per_bit[p][fam]:
                            picked = _pick_atom_from_family(atoms_per_bit[p][fam])
                            break
                if picked is None:
                    return None
                chosen[p] = picked

    # All slots must be filled.
    if any(a is None for a in chosen):
        return None
    return PerBitRule(atoms=tuple(chosen))  # type: ignore[arg-type]


def _iter_all_atoms(per_fam: dict[str, list[PerBitAtom]]) -> Iterable[PerBitAtom]:
    for fam in _SECTIONS:
        for atom in per_fam[fam]:
            yield atom


def _pick_atom_from_family(atoms: list[PerBitAtom]) -> PerBitAtom:
    # Prefer smallest operand indices (lexicographic) for deterministic output.
    return sorted(atoms, key=_atom_repr_key)[0]


def _stride_hint(
    family: str,
    left_chain: list[PerBitAtom],
    right_chain: list[PerBitAtom],
) -> tuple[int, int | None] | None:
    """Compute (operand_p_at_pos0, operand_q_at_pos0) if either the left or
    right chain is in `family` and gives us a stride to extrapolate from.
    Returns None if no hint can be inferred (e.g. Constant family)."""
    if family == "Constant" or family in _TRIPLE_FAMILIES:
        return None
    # Prefer the longer chain.
    if len(right_chain) >= len(left_chain) and right_chain:
        if right_chain[0].family == family:
            anchor = right_chain[0]
            # right_chain starts at position (N_BITS - len(right_chain)).
            anchor_pos = N_BITS - len(right_chain)
            p0 = (anchor.operands[0] - anchor_pos) % N_BITS
            q0 = (
                (anchor.operands[1] - anchor_pos) % N_BITS
                if family in _PAIR_FAMILIES
                else None
            )
            return (p0, q0)
    if left_chain and left_chain[0].family == family:
        anchor = left_chain[0]  # at position 0
        p0 = anchor.operands[0]
        q0 = anchor.operands[1] if family in _PAIR_FAMILIES else None
        return (p0, q0)
    return None


def _pick_stride_aware(
    cands: list[PerBitAtom],
    stride_hint: tuple[int, int | None] | None,
    position: int,
) -> PerBitAtom:
    """Pick an atom that matches the stride hint at `position` if available;
    otherwise the lexicographically simplest atom."""
    if stride_hint is not None:
        p0, q0 = stride_hint
        ep = (p0 + position) % N_BITS
        eq = (q0 + position) % N_BITS if q0 is not None else None
        for a in cands:
            if a.operands[0] != ep:
                continue
            if eq is not None:
                if len(a.operands) < 2 or a.operands[1] != eq:
                    continue
            return a
    return _pick_atom_from_family(cands)


def _apply_per_bit(rule: PerBitRule, x: int) -> int:
    in_bits = _bits_msb(x)
    out = 0
    for p in range(N_BITS):
        v = _evaluate_atom(rule.atoms[p], in_bits)
        out |= v << (N_BITS - 1 - p)
    return out & 0xFF


# ---- Public API -----------------------------------------------------------

def solve(pairs: Iterable[tuple[str, str]]):
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
    raise TypeError(f"unknown rule type: {type(rule).__name__}")


def describe_rule(rule) -> str:
    if isinstance(rule, InvariantRule):
        return (
            f"invariant K={rule.k} offsets={rule.offsets} "
            f"tt=0x{rule.truth_table:x} wrap={rule.wrap}"
        )
    if isinstance(rule, PerBitRule):
        parts = []
        for p, atom in enumerate(rule.atoms):
            parts.append(f"p{p}:{atom.family}{atom.operands}")
        return "per_bit  " + " ".join(parts)
    return repr(rule)
