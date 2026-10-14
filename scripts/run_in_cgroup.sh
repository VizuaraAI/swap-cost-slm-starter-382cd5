#!/usr/bin/env bash
# Run a command under a hard memory ceiling enforced by the kernel.
#
#   ./scripts/run_in_cgroup.sh 2G python -m src.bench --arm B --switch-rate 0.4
#
# This is what makes the experiment real. Without it you are measuring a
# process that is free to use all of RAM, and your "ceiling" is fiction.
#
# memory.swap.max=0 is deliberate: if the kernel is allowed to swap, you
# measure the swap subsystem instead of your own model swapping.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <ceiling, e.g. 2G> <command...>" >&2
  exit 2
fi

CEILING="$1"; shift

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run not found. On a non-systemd box, create the cgroup by hand:" >&2
  echo "  sudo mkdir -p /sys/fs/cgroup/swapbench" >&2
  echo "  echo ${CEILING} | sudo tee /sys/fs/cgroup/swapbench/memory.max" >&2
  echo "  echo 0          | sudo tee /sys/fs/cgroup/swapbench/memory.swap.max" >&2
  echo "  echo \$\$        | sudo tee /sys/fs/cgroup/swapbench/cgroup.procs" >&2
  exit 1
fi

echo "==> running under memory.max=${CEILING}, swap disabled"
exec systemd-run --user --scope \
  -p MemoryMax="${CEILING}" \
  -p MemorySwapMax=0 \
  -- "$@"
