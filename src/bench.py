"""Benchmark entry point. One invocation runs one (arm, switch rate) cell.

    python -m src.bench --config configs/default.yaml --arm B --switch-rate 0.4

Writes one JSONL row per query into results/, which analyze.py turns into the
paper's tables. Run it under scripts/run_in_cgroup.sh to get a real memory
ceiling.
"""

from __future__ import annotations

import argparse
import os
import random
import time

from . import data, evaluate, models, router as router_mod, swapper, workload
from .config import Config
from .instrument import RSSSampler, ResultWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwapBench runner")
    parser.add_argument("--config", default=os.path.join("configs", "default.yaml"))
    parser.add_argument("--arm", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--switch-rate", type=float, default=None)
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--cold-cache", action="store_true",
                        help="drop the page cache before each swap (needs root)")
    parser.add_argument("--router", default=None)
    parser.add_argument("--router-accuracy", type=float, default=1.0)
    parser.add_argument("--tag", default="", help="suffix for the output filename")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the workload and print it, load no models")
    return parser.parse_args()


def output_path(cfg: Config, arm: str, switch_rate: float, tag: str) -> str:
    out_dir = cfg.get("experiment.out_dir", "results")
    device = cfg.get("experiment.device_label", "unknown")
    name = f"arm{arm}_r{switch_rate:.2f}_{device}"
    if tag:
        name += f"_{tag}"
    return os.path.join(out_dir, f"{name}.jsonl")


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config)

    tasks = cfg.get("workload.tasks", ["code", "math"])
    n_queries = args.n_queries or cfg.get("workload.n_queries", 120)
    switch_rate = (
        args.switch_rate
        if args.switch_rate is not None
        else cfg.get("workload.switch_rates", [0.3])[0]
    )
    seed = cfg.get("experiment.seed", 0)
    rng = random.Random(seed)

    stream = workload.switch_stream(
        tasks=tasks,
        n=n_queries,
        switch_rate=switch_rate,
        rng=rng,
        burstiness=cfg.get("workload.burstiness", "bernoulli"),
    )
    realised = workload.observed_switch_rate(stream)
    runs = workload.run_lengths(stream)
    print(
        f"arm={args.arm} target_r={switch_rate:.2f} realised_r={realised:.3f} "
        f"n={len(stream)} mean_run_len={sum(runs) / len(runs):.2f}"
    )

    if args.dry_run:
        print("stream:", " ".join(t[0].upper() for t in stream))
        return 0

    print("loading datasets ...")
    pools = data.load_pools(cfg)
    for task, pool in pools.items():
        print(f"  {task}: {len(pool)} questions")

    router = router_mod.build_router(
        args.router or cfg.get("workload.router", "oracle"),
        tasks=tasks,
        seed=seed,
        accuracy=args.router_accuracy,
    )

    max_tokens = cfg.get("runtime.max_tokens", 256)
    temperature = cfg.get("runtime.temperature", 0.0)
    path = output_path(cfg, args.arm, switch_rate, args.tag)

    queries = list(workload.interleave_queries(stream, pools, rng))

    engine = swapper.engine_for_arm(cfg, args.arm, cold_cache=args.cold_cache)
    sampler = RSSSampler(hz=cfg.get("memory.rss_sample_hz", 50))

    with engine, sampler, ResultWriter(path) as writer:
        for index, query in enumerate(queries):
            routed_task = router.route(query)
            role = swapper.role_for(args.arm, routed_task)

            swap_start = time.perf_counter()
            record = engine.ensure_resident(role)
            swap_seconds = time.perf_counter() - swap_start

            handle = engine.get(role)
            prompt = models.build_prompt(routed_task, query["question"])

            gen_start = time.perf_counter()
            generated = handle.generate(prompt, max_tokens, temperature)
            gen_seconds = time.perf_counter() - gen_start

            # Scored outside the timed region so scoring never pollutes latency.
            quality = evaluate.score(query["task"], generated, query)

            row = {
                "index": index,
                "arm": args.arm,
                "device": cfg.get("experiment.device_label", "unknown"),
                "ceiling_gb": cfg.get("memory.ceiling_gb"),
                "target_switch_rate": switch_rate,
                "realised_switch_rate": realised,
                "true_task": query["task"],
                "routed_task": routed_task,
                "routed_correctly": routed_task == query["task"],
                "qid": query.get("qid"),
                "model": handle.label,
                "swapped": record is not None,
                "swap_s": round(swap_seconds, 4),
                "generate_s": round(gen_seconds, 4),
                "total_s": round(swap_seconds + gen_seconds, 4),
                "quality": quality,
                "cold_cache_requested": args.cold_cache,
                "cold_cache_effective": not engine.cache_drop_failed,
            }
            if record is not None:
                row.update(record.to_row())
            writer.write(row)

            if index % 10 == 0:
                print(
                    f"  [{index:4d}/{len(queries)}] task={query['task']:5s} "
                    f"swap={swap_seconds:6.2f}s gen={gen_seconds:6.2f}s q={quality:.0f}"
                )

        n_swaps = len(engine.swaps)
        print(f"done: {n_swaps} swaps over {len(queries)} queries -> {path}")
        if engine.cache_drop_failed:
            print(
                "WARNING: could not drop the page cache, so these 'cold' numbers "
                "are warm. Re-run with sudo, or report them as warm."
            )
        print(f"peak RSS: {sampler.peak_bytes() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
