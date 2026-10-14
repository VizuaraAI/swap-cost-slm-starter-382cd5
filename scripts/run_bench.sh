#!/usr/bin/env bash
# Sweep both arms across the switch rates in configs/default.yaml.
# This is the run that produces Table 1 and Figure 2 of the paper.
#
#   ./scripts/run_bench.sh                 # warm page cache, no ceiling
#   CEILING=2G ./scripts/run_bench.sh      # enforce a 2 GB ceiling
#   COLD=1 CEILING=2G ./scripts/run_bench.sh   # cold cache (needs sudo)
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/default.yaml}"
RATES="${RATES:-0.0 0.2 0.4 0.6 0.8 1.0}"
ARMS="${ARMS:-A B}"
CEILING="${CEILING:-}"
COLD="${COLD:-0}"

cold_flag=""
if [ "$COLD" = "1" ]; then
  cold_flag="--cold-cache"
fi

run_one () {
  local arm="$1" rate="$2"
  echo
  echo "================ arm ${arm}, switch rate ${rate} ================"
  local cmd=(python -m src.bench --config "$CONFIG" --arm "$arm"
             --switch-rate "$rate" $cold_flag)
  if [ -n "$CEILING" ]; then
    ./scripts/run_in_cgroup.sh "$CEILING" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

for arm in $ARMS; do
  for rate in $RATES; do
    # Arm A never swaps, so its latency does not depend on the switch rate.
    # Run it once at r=0.0 and reuse that row rather than burning hours.
    if [ "$arm" = "A" ] && [ "$rate" != "0.0" ]; then
      continue
    fi
    run_one "$arm" "$rate"
  done
done

echo
echo "==> all runs finished. Now analyse:"
echo "    python -m src.analyze --results results --out-dir analysis"
