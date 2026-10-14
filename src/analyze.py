"""Turn results/*.jsonl into the paper's tables and figures.

    python -m src.analyze --results results --out-dir analysis

Produces:
  table_main.csv    rows map 1:1 to Table 1 of the paper
  table_decomp.csv  rows map 1:1 to Table 2 of the paper
  breakeven.csv     r* per (device, arm pair)
  fig2_results.png  the crossover plot, regenerated from real data

The break-even rate is the number the paper exists to report:

    L_A = T_A                 resident generalist, no swap cost
    L_B = T_B + r * C         specialists, one swap per task change
    r*  = (T_A - T_B) / C

If r* > 1 the specialists win at every switch rate. If r* < 0 the generalist is
faster even with no swapping at all, which means your specialists are too slow
rather than too often swapped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

PHASE_COLUMNS = [
    "phase_evict_s",
    "phase_storage_read_s",
    "phase_page_in_s",
    "phase_warmup_s",
]


def load_results(results_dir: str) -> pd.DataFrame:
    """Read every JSONL file in results_dir into one DataFrame."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no .jsonl files in {results_dir}")
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    print(f"loaded {len(frame)} query rows from {len(paths)} files")
    return frame


def main_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (device, arm, switch rate): latency, quality, swap cost."""
    grouped = frame.groupby(["device", "arm", "target_switch_rate"], dropna=False)
    out = grouped.agg(
        n_queries=("total_s", "size"),
        n_swaps=("swapped", "sum"),
        mean_latency_s=("total_s", "mean"),
        p95_latency_s=("total_s", lambda s: s.quantile(0.95)),
        mean_generate_s=("generate_s", "mean"),
        quality=("quality", "mean"),
        peak_rss_mb=("rss_after_mb", "max"),
    ).reset_index()
    out["swap_cost_C_s"] = grouped.apply(
        lambda g: g.loc[g["swapped"] == True, "swap_s"].mean(),  # noqa: E712
        include_groups=False,
    ).values
    out["quality_pct"] = (out["quality"] * 100).round(1)
    return out.round(4)


def decomposition_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per device: the four swap phases, cold and warm, plus churn counters."""
    swaps = frame[frame["swapped"] == True].copy()  # noqa: E712
    if swaps.empty:
        return pd.DataFrame()
    present = [c for c in PHASE_COLUMNS if c in swaps.columns]
    agg: dict = {col: (col, "median") for col in present}
    agg["swap_total_s"] = ("swap_total_s", "median")
    agg["swap_p95_s"] = ("swap_total_s", lambda s: s.quantile(0.95))
    agg["n_swaps"] = ("swap_total_s", "size")
    if "bytes_read" in swaps.columns:
        agg["bytes_read_gb"] = ("bytes_read", "median")
    if "major_faults" in swaps.columns:
        agg["major_faults"] = ("major_faults", "median")
    if "psi_stall_us" in swaps.columns:
        agg["psi_stall_ms"] = ("psi_stall_us", "median")

    keys = ["device", "arm"]
    if "cold_cache_effective" in swaps.columns:
        keys.append("cold_cache_effective")
    out = swaps.groupby(keys, dropna=False).agg(**agg).reset_index()
    if "bytes_read_gb" in out.columns:
        out["bytes_read_gb"] = (out["bytes_read_gb"] / 1e9).round(2)
    if "psi_stall_ms" in out.columns:
        out["psi_stall_ms"] = (out["psi_stall_ms"] / 1e3).round(1)
    return out.round(4)


def breakeven_table(frame: pd.DataFrame) -> pd.DataFrame:
    """r* = (T_A - T_B) / C, per device.

    T_A and T_B are *service* latencies (generation only), so that swap cost
    enters exactly once, through C. Mixing swap time into T_B would double
    count it and understate r*.
    """
    rows = []
    for device, device_frame in frame.groupby("device", dropna=False):
        arm_a = device_frame[device_frame["arm"] == "A"]
        arm_b = device_frame[device_frame["arm"] == "B"]
        if arm_a.empty or arm_b.empty:
            continue
        t_a = arm_a["generate_s"].mean()
        t_b = arm_b["generate_s"].mean()
        swaps = arm_b[arm_b["swapped"] == True]  # noqa: E712
        cost = swaps["swap_s"].mean() if not swaps.empty else float("nan")
        r_star = (t_a - t_b) / cost if cost and cost > 0 else float("nan")
        rows.append(
            {
                "device": device,
                "T_A_generalist_s": round(t_a, 3),
                "T_B_specialist_s": round(t_b, 3),
                "swap_cost_C_s": round(cost, 3) if cost == cost else None,
                "r_star": round(r_star, 3) if r_star == r_star else None,
                "quality_gain_pts": round(
                    (arm_b["quality"].mean() - arm_a["quality"].mean()) * 100, 2
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_crossover(frame: pd.DataFrame, out_path: str) -> None:
    """Regenerate figure 2(b) from measured data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 4))
    for (device, arm), group in frame.groupby(["device", "arm"], dropna=False):
        curve = group.groupby("target_switch_rate")["total_s"].mean().sort_index()
        if curve.empty:
            continue
        label = f"{device} arm {arm}"
        if len(curve) == 1:
            # Arm A never swaps, so run_bench.sh measures it at one switch rate
            # only. Draw it as the horizontal line it actually is.
            axis.axhline(curve.iloc[0], ls="--", lw=1.8, color="#2f6f9f",
                         label=label)
            continue
        style = "--" if arm == "A" else "-"
        axis.plot(curve.index, curve.values, style, marker="o", label=label)
    axis.set_xlabel("Task switch rate r")
    axis.set_ylabel("Mean latency per query (s)")
    axis.set_title("Measured latency vs task switch rate")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SwapBench analysis")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()

    frame = load_results(args.results)
    os.makedirs(args.out_dir, exist_ok=True)

    tables = {
        "table_main.csv": main_table(frame),
        "table_decomp.csv": decomposition_table(frame),
        "breakeven.csv": breakeven_table(frame),
    }
    for name, table in tables.items():
        path = os.path.join(args.out_dir, name)
        table.to_csv(path, index=False)
        print(f"\n=== {name} ===")
        print(table.to_string(index=False) if not table.empty else "(empty)")
        print(f"wrote {path}")

    plot_crossover(frame, os.path.join(args.out_dir, "fig2_results.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
