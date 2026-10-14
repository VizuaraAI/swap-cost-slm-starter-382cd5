"""Quality scoring, kept separate from the latency harness on purpose.

Scoring code in the measurement loop would pollute your timings. Everything
here is cheap and pure, but call it after the timed region anyway.
"""

from __future__ import annotations

import multiprocessing
import re
from typing import Sequence

from .data import extract_final_number

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def score_math(generated: str, gold: str) -> float:
    """Exact match on the final number. 1.0 or 0.0."""
    predicted = extract_final_number(generated)
    if predicted is None:
        return 0.0
    try:
        return 1.0 if abs(float(predicted) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if predicted.strip() == str(gold).strip() else 0.0


def strip_code_fence(text: str) -> str:
    """Pull code out of a markdown fence if the model used one."""
    match = CODE_FENCE_RE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def _exec_target(code: str, tests: Sequence[str], queue) -> None:
    """Run candidate code plus asserts in a child process."""
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102 - sandboxed by process + timeout
        for test in tests:
            exec(test, namespace)  # noqa: S102
        queue.put(True)
    except BaseException:
        queue.put(False)


def score_code(generated: str, tests: Sequence[str], timeout: float = 5.0) -> float:
    """pass@1 for MBPP: does the generated code satisfy every assert?

    Runs in a separate process with a timeout, because generated code hangs,
    segfaults and calls sys.exit. Never run it in the harness process: one
    infinite loop would cost you a whole overnight run.

    This is still not a security sandbox. For untrusted models, run the whole
    harness in a container.
    """
    code = strip_code_fence(generated)
    if not code:
        return 0.0
    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_target, args=(code, list(tests), queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        return 0.0
    try:
        return 1.0 if queue.get_nowait() else 0.0
    except Exception:
        return 0.0


def score(task: str, generated: str, query: dict) -> float:
    """Dispatch to the right scorer for a task."""
    if task == "code":
        return score_code(generated, query.get("tests", []))
    if task == "math":
        return score_math(generated, query.get("answer", ""))
    raise ValueError(f"no scorer for task '{task}'")
