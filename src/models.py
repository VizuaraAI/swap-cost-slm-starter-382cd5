"""Model loading and unloading, with the phases that matter timed separately.

llama_cpp is imported lazily so that the rest of the harness (workload
generation, analysis, tests) imports and runs on any machine, including one
without the native library built.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Optional


class ModelHandle:
    """A loaded GGUF model plus the metadata we need in result rows."""

    def __init__(self, label: str, path: str, llm: Any) -> None:
        self.label = label
        self.path = path
        self.llm = llm
        self.loaded_at = time.time()

    def file_bytes(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        return out["choices"][0]["text"]

    def close(self) -> None:
        """Release native memory.

        llama_cpp frees its buffers when the Llama object is collected, so we
        drop the reference and force a collection. Without the explicit gc call
        the free can lag far enough to pollute the next phase measurement.
        """
        self.llm = None
        gc.collect()


def load_model(
    label: str,
    path: str,
    n_ctx: int = 2048,
    n_threads: int = 4,
    n_gpu_layers: int = 0,
    use_mmap: bool = True,
    use_mlock: bool = False,
    verbose: bool = False,
) -> ModelHandle:
    """Load a GGUF model. Raises FileNotFoundError with a useful hint."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"model file not found: {path}\n"
            "Run scripts/download_models.sh first, or fix models_dir in your config."
        )
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "llama-cpp-python is not installed. Run: pip install -r requirements.txt"
        ) from exc

    llm = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        verbose=verbose,
    )
    return ModelHandle(label=label, path=path, llm=llm)


def warmup(handle: ModelHandle, prompt: str = "Hello") -> float:
    """Force the first forward pass and KV cache allocation.

    This is swap phase 4. It is the phase most informal benchmarks forget, and
    on fast NVMe it can be the largest remaining term, so measure it explicitly.
    """
    start = time.perf_counter()
    handle.generate(prompt, max_tokens=1, temperature=0.0)
    return time.perf_counter() - start


def prefetch_file(path: str, chunk_bytes: int = 8 << 20) -> int:
    """Read a file sequentially to pull it into the page cache.

    Used two ways: to warm the cache deliberately for the warm-cache arm, and
    as the "storage_read" phase of a swap so that bulk read time is separated
    from mmap page-fault time.
    """
    total = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
    except OSError:
        return total
    return total


ROLE_FOR_TASK = {
    "code": "specialist_code",
    "math": "specialist_math",
}


def role_for_task(task: str) -> str:
    """Map a task label to the specialist that handles it."""
    try:
        return ROLE_FOR_TASK[task]
    except KeyError as exc:
        raise KeyError(
            f"no specialist registered for task '{task}'; "
            f"known tasks: {sorted(ROLE_FOR_TASK)}"
        ) from exc


def build_prompt(task: str, question: str) -> str:
    """Minimal chat-style prompt, kept identical across all arms.

    Keep this identical between arms or your quality comparison is confounded
    by prompt engineering rather than by model choice.
    """
    if task == "code":
        instruction = (
            "You are a Python programmer. Write a single correct Python "
            "function that solves the problem. Output only code."
        )
    else:
        instruction = (
            "You are a careful mathematician. Solve the problem step by step, "
            "then give the final numeric answer on its own last line after '####'."
        )
    return f"<|system|>\n{instruction}\n<|user|>\n{question}\n<|assistant|>\n"


def optional_int(value: Optional[int]) -> int:
    return 0 if value is None else int(value)
