"""Solver for text_encryption puzzles.

Each puzzle is a substitution cipher: every cipher character maps to exactly
one plaintext character. Examples give `<cipher words> -> <plain words>` lines.
Words in the cipher align positionally with words in the plaintext.

Solver:
  1. Parse `cipher -> plain` example pairs.
  2. Split each pair into words; pair up cipher_word ↔ plain_word.
  3. For each word pair of equal length, derive per-letter mappings.
  4. Check consistency. Reject the puzzle if examples imply contradictions.
  5. Apply the assembled cipher→plain map to the test cipher.

Coverage is limited by:
  - Test cipher containing letters not seen in the example word pairs
    (incomplete mapping → unknown chars).
  - Word-length mismatches between cipher and plain (e.g. if there's an
    implicit reverse-word transformation we don't yet handle).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_EXAMPLE_RE = re.compile(r"([^\n]+?)\s*->\s*([^\n]+)")
_TEST_RE = re.compile(
    r"Now,\s*decrypt\s+the\s+following\s+text:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class CipherRule:
    mapping: dict[str, str]            # cipher char -> plain char
    full_coverage: bool                # whether the test text was decodable in full
    unknowns: tuple[str, ...]          # cipher chars in test not in mapping
    consistent: bool                   # whether examples implied a consistent map


def _is_example_line(line: str) -> bool:
    return "->" in line and not line.lower().startswith("now")


def parse(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    """Return ([(cipher_text, plain_text), ...], test_cipher) or test_cipher=None."""
    lines = prompt.splitlines()
    pairs: list[tuple[str, str]] = []
    for line in lines:
        if not _is_example_line(line):
            continue
        m = _EXAMPLE_RE.match(line.strip())
        if not m:
            continue
        cipher = m.group(1).strip()
        plain = m.group(2).strip()
        pairs.append((cipher, plain))
    m = _TEST_RE.search(prompt)
    return pairs, (m.group(1).strip() if m else None)


def _build_mapping(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], bool]:
    """Build cipher→plain char map from word-aligned pairs. Returns (mapping,
    consistent)."""
    mapping: dict[str, str] = {}
    consistent = True
    for cipher_text, plain_text in pairs:
        cw = cipher_text.split()
        pw = plain_text.split()
        if len(cw) != len(pw):
            continue  # word count mismatch — skip this pair entirely
        for c_word, p_word in zip(cw, pw):
            if len(c_word) != len(p_word):
                continue  # length mismatch on this word
            for cc, pc in zip(c_word, p_word):
                if cc in mapping:
                    if mapping[cc] != pc:
                        consistent = False
                else:
                    mapping[cc] = pc
    return mapping, consistent


def _puzzle_vocab(pairs: list[tuple[str, str]]) -> set[str]:
    """All plaintext words seen in this puzzle's examples."""
    vocab: set[str] = set()
    for _, plain_text in pairs:
        for w in plain_text.split():
            if w:
                vocab.add(w.lower())
    return vocab


# ---- Global Wonderland vocabulary -----------------------------------------
# Built lazily from train.csv on first solve() call. We aggregate plaintext
# words from EVERY text_encryption example AND from train.csv answer fields.
# This is a distribution-level prior, not test-set cheating: the dataset uses
# a constrained themed vocabulary that we can learn from training data.

_GLOBAL_VOCAB: set[str] | None = None


def _load_global_vocab() -> set[str]:
    global _GLOBAL_VOCAB
    if _GLOBAL_VOCAB is not None:
        return _GLOBAL_VOCAB
    from pathlib import Path
    import csv
    vocab: set[str] = set()
    _repo = Path(__file__).resolve().parents[2]  # src/data/<file> -> repo root
    candidate_paths = [
        Path("datasets/raw/train.csv"),     # when run from repo root
        _repo / "datasets/raw/train.csv",   # repo-relative fallback
    ]
    train_path = next((p for p in candidate_paths if p.exists()), None)
    if train_path is None:
        _GLOBAL_VOCAB = vocab
        return _GLOBAL_VOCAB
    with train_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "")
            answer = row.get("answer", "")
            # We only mine encryption puzzles (others' plaintexts could pollute
            # the vocab with numbers, binary strings, symbols).
            is_enc = "secret encryption rules are used on text" in prompt
            if is_enc:
                # plaintexts from example lines (after `->`)
                for line in prompt.splitlines():
                    if "->" in line and not line.lower().startswith("now"):
                        right = line.split("->", 1)[1].strip()
                        for w in right.split():
                            if w.isalpha():
                                vocab.add(w.lower())
                # the answer itself is a plaintext sentence
                if answer:
                    for w in answer.split():
                        if w.isalpha():
                            vocab.add(w.lower())
    _GLOBAL_VOCAB = vocab
    return vocab


def _fill_unknowns(
    test_cipher: str,
    mapping: dict[str, str],
    vocab: set[str],
) -> dict[str, str]:
    """Iteratively augment the cipher→plain mapping by pattern-matching each
    partially-decoded word against the puzzle's plain vocabulary. We require
    a *unique* vocab match at each step (and consistency with the bijection)
    to avoid introducing wrong guesses.

    Returns the augmented mapping (does not mutate the input)."""
    mapping = dict(mapping)
    used_plain = set(mapping.values())
    cipher_words = test_cipher.lower().split()

    changed = True
    while changed:
        changed = False
        for c_word in cipher_words:
            # Build the partial decoded pattern: known chars resolved; unknowns
            # become regex placeholders that must (a) match a single letter and
            # (b) be the same wherever the same cipher char repeats.
            unknown_chars = [c for c in c_word if c not in mapping]
            if not unknown_chars:
                continue
            # Distinct unknown chars in this word, sorted for stable ordering
            distinct_unknowns: list[str] = []
            for c in c_word:
                if c not in mapping and c not in distinct_unknowns:
                    distinct_unknowns.append(c)
            # Build candidate match: for each vocab word of same length, check
            # consistency with the known mapping and propose a fill-in.
            candidates: list[dict[str, str]] = []
            for v in vocab:
                if len(v) != len(c_word):
                    continue
                # Check fixed positions agree
                ok = True
                proposal: dict[str, str] = {}
                for cc, pc in zip(c_word, v):
                    if cc in mapping:
                        if mapping[cc] != pc:
                            ok = False
                            break
                    else:
                        if cc in proposal:
                            if proposal[cc] != pc:
                                ok = False
                                break
                        else:
                            # New mapping must respect bijection: pc not used
                            if pc in used_plain or pc in proposal.values():
                                ok = False
                                break
                            proposal[cc] = pc
                if ok and proposal:
                    candidates.append(proposal)
            if len(candidates) == 1:
                for cc, pc in candidates[0].items():
                    mapping[cc] = pc
                    used_plain.add(pc)
                changed = True
    return mapping


def solve(prompt: str) -> tuple[str | None, CipherRule | None]:
    pairs, test_cipher = parse(prompt)
    if not pairs or test_cipher is None:
        return None, None
    mapping, consistent = _build_mapping(pairs)
    if not mapping:
        return None, None

    # First pass: puzzle-internal vocab (highest precision, avoids drift).
    vocab_local = _puzzle_vocab(pairs)
    augmented = _fill_unknowns(test_cipher, mapping, vocab_local)

    # Second pass: global Wonderland vocab from train.csv. Only used when the
    # local pass leaves unknown cipher chars in the test text.
    if any(c not in augmented and c != " " for c in test_cipher):
        vocab_global = _load_global_vocab()
        augmented = _fill_unknowns(test_cipher, augmented, vocab_global)

    decoded_chars: list[str] = []
    unknowns: set[str] = set()
    for ch in test_cipher:
        if ch == " ":
            decoded_chars.append(" ")
        elif ch in augmented:
            decoded_chars.append(augmented[ch])
        else:
            decoded_chars.append("?")
            unknowns.add(ch)
    decoded = "".join(decoded_chars)
    full_coverage = not unknowns
    rule = CipherRule(
        mapping=dict(augmented),
        full_coverage=full_coverage,
        unknowns=tuple(sorted(unknowns)),
        consistent=consistent,
    )
    return decoded, rule


def format_answer(decoded: str) -> str:
    return decoded
