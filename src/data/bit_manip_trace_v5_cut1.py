"""Shortened bit_manip v5 trace (Cut 1) for the v12 build.

Identical to `bit_manip_trace_v5` except it drops the raw input->output example
list, which is pure redundancy with the transposed Input/Output columns (the
columns are what every derivation line actually references). All
derivability-critical regions are untouched: per-bit elimination (the XOR
partner-solve / AND-OR 0/1-row masking), the `=> out[...]` conclusions, the
earlier-family rejections, and the stride-run verifications — i.e. exactly the
content that produced the +16.7pp bit_manip gain in v11bm.

Token impact (dev_frozen): recovers ~25% of trace length (the example list was
~30% of tokens, ~half of which overlapped with columns), reducing shared-LoRA
capacity pressure on text_encryption map construction (Q9 regression mechanism)
and the v11bm bit_manip free-run truncation (4.8%).

Contract matches v5: `generate_trace(rule, pairs, test_input) -> (trace, answer)`
or None.
"""
from __future__ import annotations

from .bit_manip_trace_v5 import generate_trace as _v5_generate_trace


def generate_trace(rule, pairs, test_input):
    return _v5_generate_trace(rule, pairs, test_input, include_examples=False)
