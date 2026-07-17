"""Derivation-first trace generator for the v3 bit_manipulation solver.

Motivation (see docs/investigations/OPEN_QUESTIONS.md Q8): the compact v4 trace
STATES each output bit's operator+indices without DERIVING them. Teacher-forced
probing showed 44/47 model-wrong covered traces first diverge at exactly those
rule-statement tokens. v5 fixes the format constraint: **every operator and bit
index in the trace is forced by a prior, self-verifying line** — unary equality,
XOR/XOR-NOT partner-solving (the partner is algebraically forced), or AND/OR
(+NOT) 0/1-row masking (operand candidates are forced by the rows where the
target column is 0 or 1). Interior bits of a stride run are emitted as +1
extrapolations that are then verified, so only run anchors carry the full
elimination.

Inspired by tonghuikang's reference `reasoners/bit_manipulation.py`: bit-column
transpose, accept-by-column-equality, and Left/Right stride runs. Dropped: his
full per-family candidate dump (too long).

Contract matches v4: `generate_trace(rule, pairs, test_input) -> (trace, answer)`
or None if a bit uses a family this generator does not derive (rare; logged by
the caller via None). The returned answer is computed from the narrated atoms,
so build-time verification (answer == ground truth) still gates inclusion.
"""
from __future__ import annotations

from itertools import combinations

from .bit_manip_solver_v3 import InvariantRule, PerBitRule, apply_rule

N_BITS = 8

# Families this generator can derive with full elimination. Order = preference
# (mirrors the solver's section order: unary/constant first, then pairs).
_PAIR_ORDER = ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT")
_SUPPORTED_PAIR = set(_PAIR_ORDER)


# ---- column helpers (columns are bit-strings down the examples) -----------

def _inv(s: str) -> str:
    return "".join("1" if c == "0" else "0" for c in s)


def _and(a: str, b: str) -> str:
    return "".join("1" if (x == "1" and y == "1") else "0" for x, y in zip(a, b))


def _or(a: str, b: str) -> str:
    return "".join("1" if (x == "1" or y == "1") else "0" for x, y in zip(a, b))


def _xor(a: str, b: str) -> str:
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def _apply_pair(fam: str, a: str, b: str) -> str:
    if fam == "AND":
        return _and(a, b)
    if fam == "OR":
        return _or(a, b)
    if fam == "XOR":
        return _xor(a, b)
    if fam == "AND-NOT":
        return _and(a, _inv(b))
    if fam == "OR-NOT":
        return _or(a, _inv(b))
    if fam == "XOR-NOT":
        return _xor(a, _inv(b))
    raise ValueError(fam)


def _columns(pairs):
    ins = [i for i, _ in pairs]
    outs = [o for _, o in pairs]
    icols = ["".join(s[p] for s in ins) for p in range(N_BITS)]
    ocols = ["".join(s[p] for s in outs) for p in range(N_BITS)]
    return icols, ocols


# ---- per-family search (returns all valid operand tuples) -----------------

def _search_family(fam: str, T: str, cols, notcols):
    """Return list of valid operand tuples for `fam` reproducing column T."""
    out = []
    if fam in ("AND", "OR", "XOR"):  # symmetric
        for i, j in combinations(range(N_BITS), 2):
            if _apply_pair(fam, cols[i], cols[j]) == T:
                out.append((i, j))
    else:  # asymmetric *-NOT: ordered pairs, second operand inverted
        base = fam.split("-")[0]
        for i in range(N_BITS):
            for j in range(N_BITS):
                if i == j:
                    continue
                if _apply_pair(base, cols[i], notcols[j]) == T:
                    out.append((i, j))
    return out


# ---- full-elimination narration for one bit (anchor) ----------------------

def _ones_rows(T):  return [k for k, c in enumerate(T) if c == "1"]
def _zero_rows(T):  return [k for k, c in enumerate(T) if c == "0"]


def _cols_eq_on(cols, rows, val):
    """Input column indices whose value is `val` on every row in `rows`."""
    return [j for j in range(N_BITS) if all(cols[j][r] == val for r in rows)]


def _derive_unary(T, cols, notcols):
    """Return (atom, lines) for a unary/constant target, or None."""
    if "1" not in T:
        return ("Constant", ("0",)), [f"  T is all-0 -> out = 0."]
    if "0" not in T:
        return ("Constant", ("1",)), [f"  T is all-1 -> out = 1."]
    for j in range(N_BITS):
        if cols[j] == T:
            return ("Identity", (j,)), [f"  T equals input column I{j} -> out = input[{j}]."]
    for j in range(N_BITS):
        if notcols[j] == T:
            return ("NOT", (j,)), [f"  T equals NOT(I{j}) -> out = NOT input[{j}]."]
    return None


def _reject_unary_line(T, cols, notcols) -> str:
    return "  No input column or its complement equals T, and T is not constant -> not unary."


def _eliminate_pair(fam, ops, T, cols, notcols):
    """Lines that force the pair atom (fam, ops) for target column T.
    Assumes (fam, ops) is valid (column matches)."""
    i, j = ops
    lines = []
    if fam in ("XOR", "XOR-NOT"):
        # Partner is algebraically forced: for XOR, Ij = T XOR Ii;
        # for XOR-NOT, NOT(Ij) = T XOR Ii  =>  Ij = NOT(T XOR Ii).
        target_inv = fam == "XOR-NOT"
        lines.append(
            f"  XOR{'-NOT' if target_inv else ''}: the partner is forced by "
            f"{'NOT(Ij)=' if target_inv else 'Ij='}T XOR Ii. Scan first operand:")
        for ii in range(N_BITS):
            partner = _xor(T, cols[ii])           # = (NOT Ij) for XOR-NOT, else Ij
            want = _inv(partner) if target_inv else partner
            hit = next((k for k in range(N_BITS) if cols[k] == want), None)
            if ii == i:
                if target_inv:
                    lines.append(
                        f"    i={ii}: T XOR I{ii}={partner}; NOT={want} = I{j} "
                        f"-> XOR(I{ii},NOT I{j})={_apply_pair(fam,cols[ii],cols[j])}=T. accepted.")
                else:
                    lines.append(
                        f"    i={ii}: T XOR I{ii}={partner} = I{j} "
                        f"-> XOR(I{ii},I{j})={_apply_pair(fam,cols[ii],cols[j])}=T. accepted.")
                break
            else:
                if hit is None:
                    lines.append(f"    i={ii}: T XOR I{ii}={partner}"
                                 f"{'; NOT='+want if target_inv else ''} -> not a column.")
                else:
                    # a competing valid first operand exists before the chosen one
                    lines.append(f"    i={ii}: matches I{hit} too; prefer the run/stride choice below.")
        return lines
    # AND / OR / AND-NOT / OR-NOT: mask-forced operands.
    base = fam.split("-")[0]
    notted = fam.endswith("-NOT")
    if base == "AND":
        rows = _ones_rows(T)
        a_cands = _cols_eq_on(cols, rows, "1")
        b_cands = _cols_eq_on(cols, rows, "0") if notted else a_cands
        lines.append(f"  {fam}: where T=1 (rows {rows}) we need "
                     f"a=1 and {'b=0' if notted else 'b=1'}.")
    else:  # OR
        rows = _zero_rows(T)
        a_cands = _cols_eq_on(cols, rows, "0")
        b_cands = _cols_eq_on(cols, rows, "1") if notted else a_cands
        lines.append(f"  {fam}: where T=0 (rows {rows}) we need "
                     f"a=0 and {'b=1' if notted else 'b=0'}.")
    lines.append(f"    a-candidates {a_cands}; b-candidates {b_cands}.")
    bexpr = f"NOT I{j}" if notted else f"I{j}"
    lines.append(f"    verify {base}(I{i},{bexpr})={_apply_pair(fam,cols[i],cols[j])}=T. accepted.")
    return lines


def _atom_expr(fam, ops) -> str:
    if fam == "Constant":
        return ops[0]
    if fam == "Identity":
        return f"input[{ops[0]}]"
    if fam == "NOT":
        return f"NOT input[{ops[0]}]"
    i, j = ops
    if fam.endswith("-NOT"):
        return f"{fam.split('-')[0]}(input[{i}], NOT input[{j}])"
    return f"{fam}(input[{i}], input[{j}])"


# ---- rule -> per-bit target atoms -----------------------------------------

def _find_atom(T, cols, notcols):
    """Smallest-first atom (preference order) reproducing column T, or None."""
    u = _derive_unary(T, cols, notcols)
    if u is not None:
        return u[0]
    for fam in _PAIR_ORDER:
        res = _search_family(fam, T, cols, notcols)
        if res:
            return (fam, res[0])
    return None


def _rule_to_atoms(rule, cols, notcols):
    """Return list of 8 (fam, ops) atoms, or None if unsupported."""
    if isinstance(rule, PerBitRule):
        atoms = []
        for a in rule.atoms:
            if a.family in ("Identity", "NOT", "Constant") or a.family in _SUPPORTED_PAIR:
                atoms.append((a.family, a.operands))
            else:
                return None  # triple/parity/popcount/NAND/NOR: not derived here
        return atoms
    # InvariantRule is handled by _invariant_atoms.
    return None


# Map a K=2 truth table (minterm index = a + 2*b) to (family, swap_operands).
# Degenerate TTs are filtered by the solver; NAND/NOR are unsupported here.
_TT2_MAP = {
    0x8: ("AND", False),       # AND(a, b)
    0xE: ("OR", False),        # OR(a, b)
    0x6: ("XOR", False),       # XOR(a, b)
    0x2: ("AND-NOT", False),   # AND(a, NOT b)
    0x4: ("AND-NOT", True),    # AND(b, NOT a)  -> operands swapped
    0xB: ("OR-NOT", False),    # OR(a, NOT b)
    0xD: ("OR-NOT", True),     # OR(b, NOT a)
    0x9: ("XOR-NOT", False),   # XNOR = XOR(a, NOT b)
}


def _reduce_padded(base, notted, o0_in, o1_in, o0, o1):
    """Reduce a pair atom when a zero-padded (out-of-range) operand makes it
    constant/unary. o0 is the first operand (X), o1 the second (Y, NOT'd when
    `notted`). *_in flags say whether that operand index is in range."""
    if o0_in and o1_in:
        return None  # not reduced
    if not o0_in and not o1_in:
        yv = 1 if notted else 0           # Y' with X=0,Y=0
        if base == "AND":
            return ("Constant", ("0",))
        return ("Constant", (str(yv),))   # OR/XOR of 0 and Y'
    if not o0_in:                          # X padded (=0), Y present
        if base == "AND":
            return ("Constant", ("0",))
        return ("NOT", (o1,)) if notted else ("Identity", (o1,))
    # Y padded, X present
    if base == "AND":
        return ("Identity", (o0,)) if notted else ("Constant", ("0",))
    if base == "OR":
        return ("Constant", ("1",)) if notted else ("Identity", (o0,))
    return ("NOT", (o0,)) if notted else ("Identity", (o0,))  # XOR


def _invariant_atoms(rule):
    """Structural per-bit atoms for an InvariantRule (k<=2), or None (k=3 /
    unsupported TT)."""
    if rule.k == 1:
        d = rule.offsets[0]
        is_not = rule.truth_table == 0b01
        atoms = []
        for p in range(N_BITS):
            idx = p + d
            if rule.wrap == "wrap":
                idx %= N_BITS
                atoms.append(("NOT" if is_not else "Identity", (idx,)))
            elif 0 <= idx < N_BITS:
                atoms.append(("NOT" if is_not else "Identity", (idx,)))
            else:
                atoms.append(("Constant", ("1" if is_not else "0",)))
        return atoms
    if rule.k == 2:
        mapped = _TT2_MAP.get(rule.truth_table)
        if mapped is None:
            return None
        fam, swap = mapped
        base = fam.split("-")[0]
        notted = fam.endswith("-NOT")
        d0, d1 = rule.offsets
        atoms = []
        for p in range(N_BITS):
            a_idx, b_idx = p + d0, p + d1
            if swap:
                a_idx, b_idx = b_idx, a_idx
            if rule.wrap == "wrap":
                atoms.append((fam, (a_idx % N_BITS, b_idx % N_BITS)))
            else:
                a_in, b_in = 0 <= a_idx < N_BITS, 0 <= b_idx < N_BITS
                if a_in and b_in:
                    atoms.append((fam, (a_idx, b_idx)))
                else:
                    atoms.append(_reduce_padded(base, notted, a_in, b_in, a_idx, b_idx))
        return atoms
    return None


# ---- stride runs ----------------------------------------------------------

def _int_ops(fam, ops) -> bool:
    """True if the atom's operands are bit indices that can be strided."""
    return fam != "Constant" and all(isinstance(o, int) for o in ops)


def _runs(atoms):
    """Group indices into maximal stride-+1 (mod 8) same-family runs.
    Constant atoms (string operands) never merge into a stride run."""
    runs = []
    p = 0
    while p < N_BITS:
        q = p
        while q + 1 < N_BITS:
            fam_q, ops_q = atoms[q]          # compare against the CURRENT bit
            if not _int_ops(fam_q, ops_q):
                break
            nf, no = atoms[q + 1]
            if nf != fam_q or len(no) != len(ops_q) or not _int_ops(nf, no):
                break
            if tuple((o + 1) % N_BITS for o in ops_q) != tuple(no):
                break
            q += 1
        runs.append((p, q))
        p = q + 1
    return runs


# ---- main ------------------------------------------------------------------

def generate_trace(rule, pairs, test_input, include_examples: bool = True):
    icols, ocols = _columns(pairs)
    notcols = [_inv(c) for c in icols]

    # Per-bit target atoms. PerBit -> solver atoms; Invariant -> structural
    # conversion, falling back to per-column re-derivation. The answer
    # cross-check below guards correctness regardless of which path is taken.
    if isinstance(rule, PerBitRule):
        atoms = _rule_to_atoms(rule, icols, notcols)
    else:
        atoms = _invariant_atoms(rule)
        if atoms is None or any(a is None for a in atoms):
            atoms = []
            for p in range(N_BITS):
                a = _find_atom(ocols[p], icols, notcols)
                if a is None:
                    atoms = None
                    break
                atoms.append(a)
    if atoms is None:
        return None

    # Answer computed from the narrated atoms; cross-check vs the solver.
    pred = "".join(str(_eval_atom(fam, ops, test_input)) for fam, ops in atoms)
    if pred != f"{apply_rule(rule, int(test_input, 2)):08b}":
        return None  # narrated generalization disagrees with solver; drop + log

    lines = []
    lines.append(f"Test input: {test_input}. I must deduce the per-output-bit "
                 f"rule from the {len(pairs)} examples, then apply it.")
    lines.append("")
    if include_examples:
        # Cut 1 (v12): this raw example list is pure redundancy with the columns
        # below (same data, transposed). Omitting it preserves every
        # derivability-critical region (elimination, conclusions, rejections,
        # run:verify) — see bit_manip_trace_v5_cut1.
        lines.append("Examples (input -> output):")
        for i, o in pairs:
            lines.append(f"{i} -> {o}")
        lines.append("")
    lines.append("Read each bit position as a column down the examples "
                 "(MSB = position 0). NOT(X)=bitwise complement.")
    lines.append("Input columns:  " + "  ".join(f"I{p}={icols[p]}" for p in range(N_BITS)))
    lines.append("Output columns: " + "  ".join(f"O{p}={ocols[p]}" for p in range(N_BITS)))
    lines.append("")
    lines.append("For each output column find the operation whose computed "
                 "column equals it (accept only on equality).")

    for (lo, hi) in _runs(atoms):
        fam, ops = atoms[lo]
        T = ocols[lo]
        lines.append(f"Bit {lo}: target O{lo}={T}.")
        if fam in ("Identity", "NOT", "Constant"):
            _, ulines = _derive_unary(T, icols, notcols)
            lines.extend(ulines)
        else:
            lines.append(_reject_unary_line(T, icols, notcols))
            # reject earlier-preference pair families (compact, mask-justified)
            for ef in _PAIR_ORDER:
                if ef == fam:
                    break
                if not _search_family(ef, T, icols, notcols):
                    lines.append(f"  {ef}: no operand pair reproduces T.")
            lines.extend(_eliminate_pair(fam, ops, T, icols, notcols))
        lines.append(f"  => out[{lo}] = {_atom_expr(fam, ops)}.")
        if hi > lo:
            lines.append(f"  Bits {lo+1}-{hi} continue this run, operands +1 (mod 8). Verify:")
            for p in range(lo + 1, hi + 1):
                f2, o2 = atoms[p]
                lines.append(f"    out[{p}]: {_verify_str(f2, o2, icols, notcols)}=O{p}"
                             f"  => {_atom_expr(f2, o2)}.")

    lines.append("")
    bits = " ".join(f"{k}:{b}" for k, b in enumerate(test_input))
    lines.append(f"Apply to test {test_input} (bits {bits}):")
    for p, (fam, ops) in enumerate(atoms):
        lines.append(f"  out[{p}]={_apply_expr(fam, ops, test_input)}={_eval_atom(fam, ops, test_input)}")
    lines.append(f"Answer bits 0..7 = {pred}")
    lines.append(f"final {pred}")
    return "\n".join(lines), pred


# ---- atom evaluation on the test input ------------------------------------

def _bit(x, i):  return int(x[i])


def _eval_atom(fam, ops, x) -> int:
    if fam == "Constant":
        return int(ops[0])
    if fam == "Identity":
        return _bit(x, ops[0])
    if fam == "NOT":
        return 1 - _bit(x, ops[0])
    i, j = ops
    a = _bit(x, i)
    b = _bit(x, j)
    base = fam.split("-")[0]
    if fam.endswith("-NOT"):
        b = 1 - b
    if base == "AND":
        return a & b
    if base == "OR":
        return a | b
    return a ^ b


def _apply_expr(fam, ops, x) -> str:
    if fam == "Constant":
        return ops[0]
    if fam == "Identity":
        return f"b{ops[0]}({_bit(x,ops[0])})"
    if fam == "NOT":
        return f"NOT b{ops[0]}(={1-_bit(x,ops[0])})"
    i, j = ops
    base = fam.split("-")[0]
    bexpr = f"NOT b{j}" if fam.endswith("-NOT") else f"b{j}"
    return f"{base}(b{i},{bexpr})"


def _verify_str(fam, ops, cols, notcols) -> str:
    if fam == "Identity":
        return f"I{ops[0]}={cols[ops[0]]}"
    if fam == "NOT":
        return f"NOT I{ops[0]}={notcols[ops[0]]}"
    if fam == "Constant":
        return f"const {ops[0]}"
    i, j = ops
    if fam.endswith("-NOT"):
        return f"{fam.split('-')[0]}(I{i},NOT I{j})={_apply_pair(fam,cols[i],cols[j])}"
    return f"{fam}(I{i},I{j})={_apply_pair(fam,cols[i],cols[j])}"
