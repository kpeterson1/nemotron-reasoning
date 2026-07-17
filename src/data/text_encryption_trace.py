"""Trace generator for text_encryption puzzles."""
from __future__ import annotations

from .text_encryption_solver import (
    parse, _build_mapping, _puzzle_vocab, _load_global_vocab, _fill_unknowns,
)


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
        "words (matched by position and length) to derive the letter mapping."
    )

    # Show one or two example pairs being decoded
    shown = 0
    for cipher_text, plain_text in pairs[:3]:
        cw = cipher_text.split()
        pw = plain_text.split()
        if len(cw) != len(pw):
            continue
        parts.append(f"Pair: '{cipher_text}' -> '{plain_text}'")
        for c_word, p_word in zip(cw, pw):
            if len(c_word) == len(p_word):
                mapping_str = ", ".join(f"{c}→{p}" for c, p in zip(c_word, p_word))
                parts.append(f"  {c_word} → {p_word}  ({mapping_str})")
        shown += 1
        if shown >= 2:
            break

    # Display the assembled mapping (sorted for stable output)
    sorted_map = sorted(base_map.items())
    map_line = ", ".join(f"{c}→{p}" for c, p in sorted_map)
    parts.append(f"Assembled cipher→plain map: {{{map_line}}}")

    # Now decode the test cipher with the base map; show unknowns; then
    # backfill from vocab and explain.
    decoded_chars = []
    unknowns = []
    for ch in test_cipher:
        if ch == " ":
            decoded_chars.append(" ")
        elif ch in base_map:
            decoded_chars.append(base_map[ch])
        else:
            decoded_chars.append("?")
            if ch not in unknowns:
                unknowns.append(ch)
    initial_decode = "".join(decoded_chars)

    parts.append(f"Decode the test cipher '{test_cipher}' with this map:")
    parts.append(f"  initial: '{initial_decode}'")

    full_map = dict(base_map)
    if unknowns:
        parts.append(f"  unknown cipher letters: {', '.join(unknowns)}")
        local_vocab = _puzzle_vocab(pairs)
        full_map = _fill_unknowns(test_cipher, base_map, local_vocab)
        if any(c not in full_map and c != " " for c in test_cipher):
            global_vocab = _load_global_vocab()
            full_map = _fill_unknowns(test_cipher, full_map, global_vocab)
        # Show only the NEW additions
        new = sorted(set(full_map) - set(base_map))
        if new:
            additions = ", ".join(f"{c}→{full_map[c]}" for c in new)
            parts.append(f"  fill unknowns from Wonderland vocabulary: {additions}")

    final_chars = []
    for ch in test_cipher:
        if ch == " ":
            final_chars.append(" ")
        elif ch in full_map:
            final_chars.append(full_map[ch])
        else:
            final_chars.append("?")
    decoded = "".join(final_chars)
    parts.append(f"Final decoded text: '{decoded}'")
    return "\n".join(parts), decoded
