# Experiment log

Copy this file for every experiment: `experiments/2026-10-21-pi5-nvme-ceiling2g.md`.

The point of the log is that in week 7, when you are writing the paper, you can
answer "where did this number come from?" without rerunning anything. A result
you cannot trace is a result you cannot publish.

The tables below map one to one onto the paper. Fill the same cells you will
paste into `main.tex`.

---

## Header

| field | value |
|---|---|
| Date | 2026-__-__ |
| Experiment ID | e.g. `exp-014` |
| Git commit | `git rev-parse --short HEAD` |
| Config file | `configs/default.yaml` |
| Device label | e.g. `pi5-nvme` |
| Device detail | board, RAM, storage medium, cooling, OS and kernel |
| Memory ceiling `M` | e.g. 2 GB, enforced with `scripts/run_in_cgroup.sh` |
| Page cache | cold / warm (and whether `drop_page_cache` actually succeeded) |
| Runtime | llama.cpp build, `n_threads`, `n_gpu_layers`, quant |
| Results file | `results/armB_r0.40_pi5-nvme.jsonl` |

## Question

One sentence. What would change your mind about the project if this run came
out the other way?

## Hypothesis

Write the prediction **before** you run it, with a number if you can.
A prediction you wrote afterwards teaches you nothing.

---

## Table 1 rows (main comparison)

Maps to Table 1 of the paper.

| Arm | Device | Peak RSS (GB) | Swap cost `C` (s) | Mean latency (s) | Quality (%) | Energy (J) | `r*` |
|---|---|---|---|---|---|---|---|
| A: generalist | | | n/a | | | | n/a |
| B: specialists | | | | | | | |

## Table 2 rows (swap decomposition)

Maps to Table 2 of the paper. Median of at least 30 swaps, cold cache.

| Device | Evict + munmap (s) | Storage read (s) | Page-in faults (s) | Warm-up TTFT (s) | Total (s) | Bytes read (GB) | Major faults (k) | Total p95 (s) |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Break-even

Equation 1 of the paper: `r* = (T_A - T_B) / C`.

| quantity | value |
|---|---|
| `T_A` generalist service latency (s) | |
| `T_B` specialist service latency (s) | |
| `C` mean swap cost (s) | |
| `r*` predicted | |
| `r*` measured from the sweep | |
| predicted vs measured agree? | |

If predicted and measured disagree by more than about 20 percent, something
outside the model is happening (queueing, thermal throttling, cache effects).
That disagreement is a finding, not a bug. Write down what you think it is.

---

## Sanity checks

Tick these before you believe any number above.

- [ ] `memory.events` shows no OOM kills for the run
- [ ] cold runs really dropped the page cache (`cold_cache_effective` is true)
- [ ] realised switch rate is within 0.05 of the target
- [ ] device was at steady-state temperature, not freshly booted
- [ ] no other heavy process was running (checked with `top`)
- [ ] at least 30 swaps contribute to each median
- [ ] quality numbers are per task, not silently averaged over an unbalanced mix

## What happened

Plots, surprises, anything that broke. Paste the console output.

## What I concluded

Two or three sentences, in plain language.

## What I will run next

One concrete next experiment. If this run killed the idea, say so plainly and
write down why. Negative results are the cheapest thing you will ever publish,
and a log full of them is what makes the final paper honest.
