"""Parse model outputs for downstream verification — placeholder."""
from __future__ import annotations

from src.evaluation.extract_answer import extract_boxed_answer


def parse_output(raw: str) -> dict[str, str]:
    """Split raw model output into thinking trace and final answer."""
    answer = extract_boxed_answer(raw)
    think = ""
    if "<think>" in raw and "</think>" in raw:
        think = raw.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return {"think": think, "answer": answer}
