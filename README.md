# swap-cost-slm-starter

**SwapBench: measuring model-swap cost under a fixed memory ceiling.**

Starter repository for Prashanth Kumar's Vizuara SLM Bootcamp research project.

## The question

Deployment guidance for small language models keeps repeating the same
trade-off: under a fixed memory ceiling you can host **one larger generalist**
that stays resident, or **two smaller specialists** that are swapped in and out
as the task changes. Specialists win on per-task quality and lose on swap
overhead.

The quality side of that claim is well documented. The swap overhead side is
asserted everywhere and, as far as we can find, measured nowhere. That is the
gap this project fills, and it is a good one: the missing quantity is small,
well defined, and reachable with hardware you already own.

The whole project collapses to one number. Let `r` be the task switch rate, the
fraction of consecutive queries that change task:

```
L_A = T_A              one resident generalist, never swaps
L_B = T_B + r * C      two specialists, one swap per task change

r* = (T_A - T_B) / C   the break-even switch rate
```

Two specialists win when `r < r*`. Nobody has measured `r*` on real devices, so
nobody can actually apply the advice. You are going to measure it.

## What is in here

```
configs/default.yaml      every knob for one experiment, in one file
src/
  config.py               YAML config with dotted-path lookup
  data.py                 MBPP and GSM8K loaders (downloads public data)
  models.py               GGUF load / unload / warm-up, timed by phase
  swapper.py              the swap engine: residency policy per arm  <- the core
  instrument.py           phase timers, RSS sampler, cgroup and PSI counters
  workload.py             task streams with a controlled switch rate r
  router.py               oracle, keyword and noisy-oracle routers
  evaluate.py             pass@1 for code, exact match for math
  bench.py                run one (arm, switch rate) cell  -> results/*.jsonl
  analyze.py              results -> the paper's tables and the crossover plot
scripts/
  setup.sh                virtualenv and dependencies
  download_models.sh      fetch the three GGUF models (about 4.5 GB)
  run_in_cgroup.sh        enforce the memory ceiling with cgroup v2
  run_bench.sh            sweep both arms across all switch rates
experiments/
  log_template.md         one log per experiment, rows map to the paper's tables
```

No data or model files are committed. The loaders download public datasets and
`download_models.sh` fetches the weights.

## Setup

```bash
./scripts/setup.sh
source .venv/bin/activate
./scripts/download_models.sh
```

On a Jetson or a CUDA laptop, rebuild the runtime with GPU support and raise
`runtime.n_gpu_layers` in `configs/default.yaml`:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## The first three things to run

**1. Check the workload generator without touching a model.** Instant, no
weights needed, and it shows you what the experiment actually varies.

```bash
python -m src.bench --arm B --switch-rate 0.4 --dry-run
```

You should see a stream of C and M labels and a realised switch rate close to
0.40. If the realised rate is far from the target, stop and fix that first:
every later number depends on it.

**2. Run one real cell.** Start with the laptop, warm cache, no ceiling. This
is your smoke test, not a result.

```bash
python -m src.bench --arm B --switch-rate 0.4 --n-queries 20
```

Watch the per-swap timings print. Your first real question: which of the four
phases dominates on this machine?

**3. Run the sweep under a real ceiling and analyse it.**

```bash
CEILING=2G ./scripts/run_bench.sh
python -m src.analyze --results results --out-dir analysis
```

That prints the two tables from the paper and writes `analysis/breakeven.csv`
with your first measured `r*`.

## Three ways to fool yourself

Worth reading before you trust any number.

1. **A warm page cache.** The second time you load a model it is already in
   RAM and the "swap" costs almost nothing. That is a real deployment case, but
   it is a *different* measurement from a cold swap. Always record which one you
   ran. `--cold-cache` needs root; if it cannot drop the cache the harness tells
   you so rather than quietly reporting warm numbers as cold.
2. **A ceiling that is not enforced.** Setting a number in the config does
   nothing on its own. Only `scripts/run_in_cgroup.sh` makes the kernel enforce
   it. Without it you are measuring a process free to use all of RAM.
3. **A confounded quality gap.** If your specialists are badly fine-tuned, Arm B
   looks bad for reasons that have nothing to do with swapping. Report
   specialist quality separately, always.

## Where this goes

`../paper/main.tex` is a placeholder draft of the paper this repo is meant to
produce. Its tables have `XX.X` where your numbers go, and
`experiments/log_template.md` has the same rows so you can paste them straight
across. Replace the placeholders one run at a time; by the time the tables are
full, so is the paper.
