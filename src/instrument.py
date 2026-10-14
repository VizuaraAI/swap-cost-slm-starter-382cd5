"""Instrumentation: the part of this project that is actually novel.

Everything here reads a counter before and after an operation and reports the
delta. Three rules learned the hard way:

1. Never report one opaque "load time". Time each phase separately, otherwise
   you cannot tell a practitioner which term to attack.
2. Always record whether the page cache was cold. A warm-cache number is a
   different measurement, not a noisy version of the same one.
3. Record the counters even when they are unavailable (None), so that a missing
   value in analysis is visibly missing rather than silently zero.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from typing import Any, Optional

import psutil

CGROUP_ROOT = "/sys/fs/cgroup"
PSI_MEMORY = "/proc/pressure/memory"

SWAP_PHASES = ("evict", "storage_read", "page_in", "warmup")


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def read_cgroup_memory_stat() -> dict[str, int]:
    """Parse memory.stat for the current cgroup, flat key/value pairs.

    Returns an empty dict off cgroup v2 (for example on macOS), which keeps the
    harness runnable on a laptop while you develop.
    """
    own = _read_text("/proc/self/cgroup")
    if own is None:
        return {}
    # cgroup v2 lines look like "0::/user.slice/....scope"
    rel = ""
    for line in own.splitlines():
        parts = line.split(":")
        if len(parts) == 3 and parts[1] == "":
            rel = parts[2].strip().lstrip("/")
            break
    text = _read_text(os.path.join(CGROUP_ROOT, rel, "memory.stat"))
    if text is None:
        text = _read_text(os.path.join(CGROUP_ROOT, "memory.stat"))
    if text is None:
        return {}
    stats: dict[str, int] = {}
    for line in text.splitlines():
        bits = line.split()
        if len(bits) == 2 and bits[1].isdigit():
            stats[bits[0]] = int(bits[1])
    return stats


def read_psi_memory() -> Optional[float]:
    """Total memory stall time in microseconds, or None if PSI is unavailable."""
    text = _read_text(PSI_MEMORY)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("full"):
            for field in line.split():
                if field.startswith("total="):
                    return float(field.split("=", 1)[1])
    return None


def read_block_read_bytes() -> Optional[int]:
    """Bytes read at the block layer by this process, if the kernel exposes it."""
    try:
        counters = psutil.Process().io_counters()
    except (psutil.Error, AttributeError):
        return None
    return int(getattr(counters, "read_bytes", 0))


def current_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def drop_page_cache() -> bool:
    """Drop the system page cache. Needs root; returns False if not permitted.

    This is what separates a cold measurement from a warm one. If it returns
    False your "cold" numbers are not cold, so the harness records the result.
    """
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as handle:
            handle.write("3\n")
        return True
    except OSError:
        return False


class RSSSampler:
    """Background thread sampling resident set size at a fixed rate.

    The peak alone hides the story. What you want for the paper is the sawtooth
    of alternating residency, which only a time series shows.
    """

    def __init__(self, hz: float = 50.0) -> None:
        self.interval = 1.0 / max(hz, 1.0)
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        start = time.perf_counter()
        while not self._stop.is_set():
            self.samples.append((time.perf_counter() - start, current_rss_bytes()))
            self._stop.wait(self.interval)

    def __enter__(self) -> "RSSSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def peak_bytes(self) -> int:
        return max((rss for _, rss in self.samples), default=0)


class PhaseTimer:
    """Accumulates named phase durations for a single swap.

    Usage:
        timer = PhaseTimer()
        with timer.phase("storage_read"):
            ...
    """

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    def phase(self, name: str) -> "_PhaseContext":
        return _PhaseContext(self, name)

    def record(self, name: str, seconds: float) -> None:
        self.durations[name] = self.durations.get(name, 0.0) + seconds

    def total(self) -> float:
        return sum(self.durations.values())


class _PhaseContext:
    def __init__(self, owner: PhaseTimer, name: str) -> None:
        self.owner = owner
        self.name = name
        self.start = 0.0

    def __enter__(self) -> "_PhaseContext":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.owner.record(self.name, time.perf_counter() - self.start)


@dataclasses.dataclass
class SwapRecord:
    """One model swap, fully described. This is the unit of the whole paper."""

    from_model: Optional[str]
    to_model: str
    cold_cache: bool
    phases: dict[str, float] = dataclasses.field(default_factory=dict)
    bytes_read: Optional[int] = None
    major_faults: Optional[int] = None
    psi_stall_us: Optional[float] = None
    rss_after_bytes: int = 0

    @property
    def total_seconds(self) -> float:
        return sum(self.phases.values())

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "from_model": self.from_model,
            "to_model": self.to_model,
            "cold_cache": self.cold_cache,
            "swap_total_s": round(self.total_seconds, 4),
            "bytes_read": self.bytes_read,
            "major_faults": self.major_faults,
            "psi_stall_us": self.psi_stall_us,
            "rss_after_mb": round(self.rss_after_bytes / 1e6, 1),
        }
        for phase in SWAP_PHASES:
            row[f"phase_{phase}_s"] = round(self.phases.get(phase, 0.0), 4)
        return row


class ResultWriter:
    """Append-only JSONL writer. One line per query, flushed immediately.

    Flushing every line matters: these runs take hours on a Pi and get killed.
    You want the partial results.
    """

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
