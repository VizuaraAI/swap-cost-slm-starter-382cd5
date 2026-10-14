"""Routing a query to a task label.

Start with the oracle. It isolates swap cost from routing error, which is what
you want for the headline result: if you use a real classifier from day one you
cannot tell whether a bad number came from swapping or from a misroute.

Then sweep router accuracy with NoisyOracleRouter to answer the practitioner's
real question: how good does my classifier have to be before two specialists
stop being worth it?
"""

from __future__ import annotations

import random
from typing import Protocol, Sequence


class Router(Protocol):
    def route(self, query: dict) -> str:
        """Return a task label for this query."""
        ...


class OracleRouter:
    """Uses the ground-truth task label. Upper bound on routing quality."""

    name = "oracle"

    def route(self, query: dict) -> str:
        return query["task"]


class KeywordRouter:
    """A deliberately simple real router: keyword counts, no training.

    Good enough to be an honest baseline and cheap enough to run on a Pi
    without competing for memory with the model under test, which a neural
    router would.
    """

    name = "keyword"

    CODE_HINTS = (
        "function", "python", "write a program", "list", "string", "array",
        "return", "sort", "dictionary", "tuple", "def ",
    )
    MATH_HINTS = (
        "how many", "how much", "total", "cost", "each", "left", "percent",
        "average", "sum of", "altogether", "spent",
    )

    def __init__(self, default: str = "math") -> None:
        self.default = default

    def route(self, query: dict) -> str:
        text = (query.get("question") or "").lower()
        code_score = sum(1 for hint in self.CODE_HINTS if hint in text)
        math_score = sum(1 for hint in self.MATH_HINTS if hint in text)
        if code_score > math_score:
            return "code"
        if math_score > code_score:
            return "math"
        return self.default


class NoisyOracleRouter:
    """Oracle that misroutes with probability 1 - accuracy.

    This is the ablation knob. A misroute both lowers quality and can trigger
    an unnecessary swap, so it hits both axes of the trade-off at once.
    """

    name = "noisy-oracle"

    def __init__(self, accuracy: float, tasks: Sequence[str], seed: int = 0) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
        self.accuracy = accuracy
        self.tasks = list(tasks)
        self.rng = random.Random(seed)

    def route(self, query: dict) -> str:
        true_task = query["task"]
        if self.rng.random() < self.accuracy or len(self.tasks) < 2:
            return true_task
        wrong = [task for task in self.tasks if task != true_task]
        return self.rng.choice(wrong)


def build_router(name: str, tasks: Sequence[str], seed: int = 0, accuracy: float = 1.0):
    """Factory used by bench.py so the config can name a router by string."""
    name = (name or "oracle").lower()
    if name == "oracle":
        return OracleRouter()
    if name == "keyword":
        return KeywordRouter()
    if name in ("noisy", "noisy-oracle"):
        return NoisyOracleRouter(accuracy=accuracy, tasks=tasks, seed=seed)
    raise ValueError(f"unknown router '{name}'")
