"""Trace generator v5 for text_encryption — character-by-character test decode.

Motivation (Q9 / docs/Q9_TEXT_ENC_LEARNING_GAP.md): the v4 trace emits the final
decode as ONE free-form string (`Final decoded text: '...'`), and the model
learns to ignore its own cipher map and emit a fluent-English sentence instead
of substituting character by character (emit==mechanical: 0/47 misses vs 81% of
correct cases). The structural fix, borrowed from Huikang's cipher reasoner, is
to gate EVERY output character on a `cipher → plain` lookup line emitted
immediately before that character is produced. The model cannot write the answer
without first committing to each per-character substitution, so the fluent-
English leap is structurally suppressed.

Single-variable discipline vs v4 (see SESSION_LOG / reviewer adjustments):
  - The example-derivation region is BYTE-IDENTICAL to v4 (`Pair:` header with
    the puzzle's own ASCII `->`, then `  cw → pw  (c→p, ...)` with Unicode →).
    The example region is not where the model fails, so we minimize the diff.
  - The assembled-map line is DROPPED: once decode is character-by-character,
    each lookup restates the mapping, and a standalone map line is just a second
    place for the model to copy a wrong map from.
  - Unknown-letter resolution stays a single COMPACT line (matched Wonderland
    word + the new mapping(s) it implies). We deliberately do NOT teach
    candidate elimination here — that is a separate hypothesis.
  - Arrow glyph matches v4 EXACTLY to avoid a silent ASCII/Unicode confound:
    ASCII `->` only in the `Pair:` header (echoing the puzzle); Unicode `→`
    (U+2192) for every derived mapping, including the per-character lookups.

The decoded answer is computed with the SAME two-pass logic as
text_encryption_solver.solve(), so v5's answer is byte-identical to v4's; only
the trace *prose* changes. Verified: v5 decode == solver decode on all 83
dev_frozen text_encryption problems.
"""
from __future__ import annotations

from .text_encryption_solver import (
    parse, _build_mapping, _puzzle_vocab, _load_global_vocab, _fill_unknowns,
)


def _full_map(pairs, test_cipher) -> dict[str, str]:
    """Reproduce solve()'s augmented map: base -> local vocab -> global vocab."""
    base, _ = _build_mapping(pairs)
    full = _fill_unknowns(test_cipher, base, _puzzle_vocab(pairs))
    if any(c not in full and c != " " for c in test_cipher):
        full = _fill_unknowns(test_cipher, full, _load_global_vocab())
    return full


def generate_trace(prompt: str) -> tuple[str, str] | None:
    pairs, test_cipher = parse(prompt)
    if not pairs or test_cipher is None:
        return None
    base_map, _ = _build_mapping(pairs)
    if not base_map:
        return None

    parts: list[str] = []
    parts.append(
        "This is a substitution cipher. I'll align cipher words with plain "
        "words (matched by position and length) to derive the letter mapping, "
        "then decode the test text one character at a time, looking up each "
        "cipher letter before writing the plain letter."
    )

    # --- Example derivation (BYTE-IDENTICAL to v4) ----------------------------
    shown = 0
    for cipher_text, plain_text in pairs[:3]:
        cw, pw = cipher_text.split(), plain_text.split()
        if len(cw) != len(pw):
            continue
        parts.append(f"Pair: '{cipher_text}' -> '{plain_text}'")
        for c_word, p_word in zip(cw, pw):
            if len(c_word) == len(p_word):
                mp = ", ".join(f"{c}→{p}" for c, p in zip(c_word, p_word))
                parts.append(f"  {c_word} → {p_word}  ({mp})")
        shown += 1
        if shown >= 2:
            break

    # (assembled-map line intentionally omitted — see module docstring)

    # --- Per-character test decode (the load-bearing change) ------------------
    full_map = _full_map(pairs, test_cipher)
    parts.append(f"Now decode the test text '{test_cipher}' word by word.")

    decoded_words: list[str] = []
    for c_word in test_cipher.split():
        new_letters = [c for c in c_word if c not in base_map]
        if new_letters:
            # Compact unknown resolution: the decoded word is the unique
            # Wonderland-vocab match; report it and the new mappings it implies.
            resolved = "".join(full_map.get(c, "?") for c in c_word)
            additions = ", ".join(
                f"{c}→{full_map[c]}" for c in dict.fromkeys(new_letters)
                if c in full_map
            )
            known = "".join(base_map.get(c, "_") for c in c_word)
            if additions:
                parts.append(
                    f"Word '{c_word}': known letters give '{known}'; the only "
                    f"Wonderland word of length {len(c_word)} matching that is "
                    f"'{resolved}', so {additions}."
                )
            else:
                parts.append(
                    f"Word '{c_word}': known letters give '{known}' (unresolved)."
                )
        else:
            parts.append(f"Word '{c_word}':")
        # Character-by-character: lookup line precedes each emitted char.
        chars: list[str] = []
        for c in c_word:
            p = full_map.get(c, "?")
            parts.append(f"  {c} → {p}")
            chars.append(p)
        parts.append(f"  = {''.join(chars)}")
        decoded_words.append("".join(chars))

    decoded = " ".join(decoded_words)
    parts.append(f"Final decoded text: '{decoded}'")
    return "\n".join(parts), decoded
