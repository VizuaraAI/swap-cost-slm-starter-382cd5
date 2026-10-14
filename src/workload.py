"""Task-labelled request streams with a controlled task switch rate.

The switch rate r is the independent variable of the whole paper, so generating
it correctly matters more than it looks.

Two generators:

  bernoulli  each step independently switches task with probability r.
             Gives exactly the target marginal switch rate, geometric run
             lengths, no extra structure.

  geometric  same marginal r, but switches arrive in bursts, so run lengths are
             over-dispersed. Use this to show that r alone does not determine
             cost: task *locality* does. That ablation is one of the more
             interesting results available to you, because it breaks the naive
             cost model.
"""

from __future__ import annotations

import random
from typing import Iterator, Sequence


def switch_stream(
    tasks: Sequence[str],
    n: int,
    switch_rate: float,
    rng: random.Random,
    burstiness: str = "bernoulli",
) -> list[str]:
    """Return a list of n task labels with the given marginal switch rate."""
    if not tasks:
        raise ValueError("tasks must be non-empty")
    if not 0.0 <= switch_rate <= 1.0:
        raise ValueError(f"switch_rate must be in [0, 1], got {switch_rate}")
    if n <= 0:
        return []

    if burstiness == "bernoulli":
        return _bernoulli_stream(tasks, n, switch_rate, rng)
    if burstiness == "geometric":
        return _bursty_stream(tasks, n, switch_rate, rng)
    raise ValueError(f"unknown burstiness '{burstiness}'")


def _bernoulli_stream(
    tasks: Sequence[str], n: int, r: float, rng: random.Random
) -> list[str]:
    current = rng.randrange(len(tasks))
    out = [tasks[current]]
    for _ in range(n - 1):
        if rng.random() < r and len(tasks) > 1:
            choices = [i for i in range(len(tasks)) if i != current]
            current = rng.choice(choices)
        out.append(tasks[current])
    return out


def _bursty_stream(
    tasks: Sequence[str], n: int, r: float, rng: random.Random
) -> list[str]:
    """Same expected switch rate, but switches cluster.

    We alternate between a "sticky" regime and a "churn" regime, spending equal
    time in each, so the mean switch probability is (sticky_p + churn_p) / 2.
    To preserve the target marginal rate that mean must equal r, hence
    sticky_p + churn_p = 2r. We spread them as far apart as the [0, 1] bounds
    allow, which maximises burstiness without distorting the marginal rate.
    """
    churn_p = min(1.0, 1.75 * r)
    sticky_p = min(max(2.0 * r - churn_p, 0.0), churn_p)
    current = rng.randrange(len(tasks))
    out = [tasks[current]]
    sticky = True
    regime_left = _regime_length(rng)
    for _ in range(n - 1):
        if regime_left <= 0:
            sticky = not sticky
            regime_left = _regime_length(rng)
        regime_left -= 1
        p = sticky_p if sticky else churn_p
        if rng.random() < p and len(tasks) > 1:
            choices = [i for i in range(len(tasks)) if i != current]
            current = rng.choice(choices)
        out.append(tasks[current])
    return out


def _regime_length(rng: random.Random) -> int:
    return max(1, int(rng.gauss(10, 3)))


def observed_switch_rate(stream: Sequence[str]) -> float:
    """Measured switch rate of a stream. Always log this next to the target.

    Targets and realisations differ on short streams, and reviewers will ask.
    """
    if len(stream) < 2:
        return 0.0
    switches = sum(1 for a, b in zip(stream, stream[1:]) if a != b)
    return switches / (len(stream) - 1)


def run_lengths(stream: Sequence[str]) -> list[int]:
    """Lengths of consecutive same-task runs, the locality summary statistic."""
    if not stream:
        return []
    lengths = [1]
    for prev, cur in zip(stream, stream[1:]):
        if cur == prev:
            lengths[-1] += 1
        else:
            lengths.append(1)
    return lengths


def interleave_queries(
    stream: Sequence[str], pools: dict[str, Sequence[dict]], rng: random.Random
) -> Iterator[dict]:
    """Attach a concrete question to each task label in the stream.

    Samples with replacement from each task's pool so that a long stream does
    not run out of questions. Yields dicts with task, question and answer.
    """
    for task in stream:
        pool = pools.get(task)
        if not pool:
            raise KeyError(f"no question pool for task '{task}'")
        item = pool[rng.randrange(len(pool))]
        yield {"task": task, **item}
