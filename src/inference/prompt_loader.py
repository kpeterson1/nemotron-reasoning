"""Load a prompt YAML and return a templating function."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml


def load_prompt(path: Path) -> Callable[..., str]:
    """Return a function that formats a problem string into the full prompt.

    For `task_typed` prompts, the returned function takes (problem, task_type).
    For all other prompts, it takes (problem) plus optional kwargs (e.g.
    `few_shot_block` for the few-shot template).
    """
    spec = yaml.safe_load(Path(path).read_text())
    if "templates" in spec:
        templates = spec["templates"]

        def render(problem: str, task_type: str) -> str:
            if task_type not in templates:
                raise KeyError(f"task_type {task_type!r} not in {sorted(templates)}")
            return templates[task_type]["template"].format(problem=problem)

        return render

    template = spec["template"]

    def render_simple(problem: str, **kwargs: str) -> str:
        return template.format(problem=problem, **kwargs)

    return render_simple
