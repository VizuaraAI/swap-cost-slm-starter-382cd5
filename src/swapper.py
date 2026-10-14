"""The swap engine. This is the heart of the experiment.

Policy per arm:
  A  resident generalist      one model, loaded once, never swapped
  B  swapped specialists      exactly one specialist resident, swap on task change
  C  shared base plus adapter one base resident, swap only the adapter
  D  partial residency        both mapped with mmap, kernel pages on demand

Every swap in arm B or C produces exactly one SwapRecord with four timed
phases. That record is the unit of analysis for the whole paper.
"""

from __future__ import annotations

from typing import Optional

from . import models
from .config import Config
from .instrument import (
    PhaseTimer,
    SwapRecord,
    current_rss_bytes,
    drop_page_cache,
    read_block_read_bytes,
    read_cgroup_memory_stat,
    read_psi_memory,
)


class SwapEngine:
    """Holds at most `capacity` models resident and swaps the rest.

    capacity=1 is the interesting case and the default: it is what a hard
    memory ceiling imposes when two specialists do not fit side by side.
    """

    def __init__(self, cfg: Config, capacity: int = 1, cold_cache: bool = False) -> None:
        self.cfg = cfg
        self.capacity = capacity
        self.cold_cache = cold_cache
        self.resident: dict[str, models.ModelHandle] = {}
        self._order: list[str] = []  # LRU, oldest first
        self.swaps: list[SwapRecord] = []
        self._cache_drop_failed = False

    # ---------------------------------------------------------------- loading

    def _runtime_kwargs(self) -> dict[str, object]:
        return {
            "n_ctx": self.cfg.get("runtime.n_ctx", 2048),
            "n_threads": self.cfg.get("runtime.n_threads", 4),
            "n_gpu_layers": self.cfg.get("runtime.n_gpu_layers", 0),
            "use_mmap": self.cfg.get("runtime.use_mmap", True),
            "use_mlock": self.cfg.get("runtime.use_mlock", False),
        }

    def _evict_one(self, timer: PhaseTimer) -> Optional[str]:
        """Evict the least recently used resident model."""
        if not self._order:
            return None
        victim_role = self._order.pop(0)
        victim = self.resident.pop(victim_role)
        with timer.phase("evict"):
            victim.close()
        return victim.label

    def ensure_resident(self, role: str) -> Optional[SwapRecord]:
        """Make `role` resident, swapping something out if needed.

        Returns a SwapRecord if a load happened, or None if it was already
        resident (a cache hit, which costs nothing and is the whole point of
        low task switch rates).
        """
        if role in self.resident:
            # Touch for LRU and report no swap.
            self._order.remove(role)
            self._order.append(role)
            return None

        timer = PhaseTimer()
        path = self.cfg.model_path(role)
        label = self.cfg.model_label(role)

        evicted_label: Optional[str] = None
        while len(self.resident) >= self.capacity:
            evicted_label = self._evict_one(timer)

        if self.cold_cache:
            if not drop_page_cache():
                self._cache_drop_failed = True

        stat_before = read_cgroup_memory_stat()
        bytes_before = read_block_read_bytes()
        psi_before = read_psi_memory()

        # Phase 2: pull the file off storage into the page cache. Doing this
        # explicitly is what lets us separate bulk read cost from the mmap
        # page-fault cost that follows.
        with timer.phase("storage_read"):
            models.prefetch_file(path)

        # Phase 3: the actual load. With mmap on, most of this is page faults
        # against a now-warm cache plus llama.cpp's own setup.
        with timer.phase("page_in"):
            handle = models.load_model(label=label, path=path, **self._runtime_kwargs())

        # Phase 4: first forward pass, KV cache allocation, time to first token.
        warm_seconds = models.warmup(handle)
        timer.record("warmup", warm_seconds)

        self.resident[role] = handle
        self._order.append(role)

        stat_after = read_cgroup_memory_stat()
        bytes_after = read_block_read_bytes()
        psi_after = read_psi_memory()

        record = SwapRecord(
            from_model=evicted_label,
            to_model=label,
            cold_cache=self.cold_cache and not self._cache_drop_failed,
            phases=dict(timer.durations),
            bytes_read=_delta(bytes_before, bytes_after),
            major_faults=_delta(
                stat_before.get("pgmajfault"), stat_after.get("pgmajfault")
            ),
            psi_stall_us=_delta(psi_before, psi_after),
            rss_after_bytes=current_rss_bytes(),
        )
        self.swaps.append(record)
        return record

    def get(self, role: str) -> models.ModelHandle:
        """Resident handle for a role. Call ensure_resident first."""
        if role not in self.resident:
            raise RuntimeError(f"model role '{role}' is not resident")
        return self.resident[role]

    def close(self) -> None:
        for handle in self.resident.values():
            handle.close()
        self.resident.clear()
        self._order.clear()

    @property
    def cache_drop_failed(self) -> bool:
        """True if a cold trial could not actually drop the cache (not root).

        Surfaced in the results so you never publish a "cold" number that was
        quietly warm.
        """
        return self._cache_drop_failed

    def __enter__(self) -> "SwapEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    """Difference of two counters, propagating None rather than guessing zero."""
    if before is None or after is None:
        return None
    return after - before


def engine_for_arm(cfg: Config, arm: str, cold_cache: bool = False) -> SwapEngine:
    """Build the engine with the residency policy for the given arm."""
    arm = arm.upper()
    if arm == "A":
        # One generalist, always resident. Capacity 1 is enough and no swap
        # ever fires because only one role is ever requested.
        return SwapEngine(cfg, capacity=1, cold_cache=cold_cache)
    if arm == "B":
        return SwapEngine(cfg, capacity=1, cold_cache=cold_cache)
    if arm == "C":
        # Shared base stays resident; adapters are swapped. Until you add
        # adapter files this behaves as a single resident base, which is the
        # right lower bound to measure first.
        return SwapEngine(cfg, capacity=1, cold_cache=cold_cache)
    if arm == "D":
        # Both specialists mapped at once; the kernel decides what to page.
        return SwapEngine(cfg, capacity=2, cold_cache=cold_cache)
    raise ValueError(f"unknown arm '{arm}'; expected one of A, B, C, D")


def role_for(arm: str, task: str) -> str:
    """Which model role serves `task` under `arm`."""
    arm = arm.upper()
    if arm == "A":
        return "generalist"
    if arm in ("B", "C", "D"):
        return models.role_for_task(task)
    raise ValueError(f"unknown arm '{arm}'")
