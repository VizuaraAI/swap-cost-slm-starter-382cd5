"""Dataset loading. Downloads public data, stores nothing large in the repo.

Two tasks, chosen because they pull a small model's capacity in different
directions, so specialisation produces a real quality gap:

  code  MBPP, scored by pass@1
  math  GSM8K, scored by exact match on the final number

`datasets` is imported lazily so the harness imports without it installed.
"""

from __future__ import annotations

import re
from typing import Any

GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _load_hf(name: str, split: str, config: str | None = None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'datasets' package is required. Run: pip install -r requirements.txt"
        ) from exc
    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)


def load_code_pool(
    dataset: str = "google-research-datasets/mbpp",
    split: str = "test",
    limit: int = 60,
) -> list[dict]:
    """MBPP problems as {question, answer, tests}.

    `answer` is the reference solution (used only for inspection, never shown
    to the model). `tests` are the assert statements used for pass@1.
    """
    rows = _load_hf(dataset, split)
    pool: list[dict] = []
    for row in rows:
        tests = row.get("test_list") or []
        prompt = row.get("text") or row.get("prompt") or ""
        if not prompt or not tests:
            continue
        pool.append(
            {
                "qid": f"mbpp-{row.get('task_id')}",
                "question": prompt.strip(),
                "answer": (row.get("code") or "").strip(),
                "tests": list(tests),
            }
        )
        if len(pool) >= limit:
            break
    return pool


def load_math_pool(
    dataset: str = "openai/gsm8k",
    split: str = "test",
    limit: int = 60,
) -> list[dict]:
    """GSM8K problems as {question, answer} with answer the final number."""
    rows = _load_hf(dataset, split, config="main")
    pool: list[dict] = []
    for index, row in enumerate(rows):
        gold = extract_final_number(row.get("answer", ""))
        if gold is None:
            continue
        pool.append(
            {
                "qid": f"gsm8k-{index}",
                "question": (row.get("question") or "").strip(),
                "answer": gold,
                "tests": [],
            }
        )
        if len(pool) >= limit:
            break
    return pool


def extract_final_number(text: str) -> str | None:
    """Pull the gold answer out of a GSM8K solution, or a model's output.

    Tries the '#### x' convention first, then falls back to the last number in
    the text, which is what a model usually produces when it ignores the format
    instruction.
    """
    match = GSM8K_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).replace(",", "").rstrip(".")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    if not numbers:
        return None
    return numbers[-1].replace(",", "").rstrip(".")


def load_pools(cfg: Any) -> dict[str, list[dict]]:
    """Build the per-task question pools described by the config."""
    n = cfg.get("data.n_per_task", 60)
    return {
        "code": load_code_pool(
            dataset=cfg.get("data.code_dataset", "google-research-datasets/mbpp"),
            split=cfg.get("data.code_split", "test"),
            limit=n,
        ),
        "math": load_math_pool(
            dataset=cfg.get("data.math_dataset", "openai/gsm8k"),
            split=cfg.get("data.math_split", "test"),
            limit=n,
        ),
    }
