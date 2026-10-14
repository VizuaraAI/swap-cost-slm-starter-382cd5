# CLAUDE.md

Instructions for the coding agent working in this repository with Prashanth.

## Project goal

Measure what on-device deployment guidance only asserts: the cost of swapping
small language models under a fixed memory ceiling, and the switch rate at
which two swapped specialists stop beating one resident generalist.

The deliverable is a workshop paper. The headline result is a single number per
device, the break-even switch rate:

```
r* = (T_A - T_B) / C
```

where `T_A` is the generalist's service latency, `T_B` the specialist's, and
`C` the mean end-to-end swap cost. Everything in this repo exists to measure
`T_A`, `T_B` and `C` honestly.

**This is a measurement paper, not a systems paper.** We are not trying to make
swapping fast. We are trying to report what it actually costs, decomposed into
phases a practitioner can act on. When a change would improve performance but
make the measurement less faithful, the measurement wins.

## Layout

| path | role |
|---|---|
| `src/swapper.py` | the core. Residency policy per arm, produces one `SwapRecord` per swap |
| `src/instrument.py` | all counters and timers. `SwapRecord` is the unit of analysis |
| `src/models.py` | GGUF load / unload / warm-up |
| `src/workload.py` | task streams with a controlled switch rate `r` |
| `src/bench.py` | entry point, one (arm, switch rate) cell per invocation |
| `src/analyze.py` | JSONL results into the paper's tables |
| `configs/default.yaml` | every experimental knob |
| `experiments/` | one markdown log per run, rows map to the paper's tables |

Arms: **A** resident generalist, **B** swapped specialists, **C** shared base
plus LoRA adapters, **D** partial residency via mmap.

## Conventions that matter

- **Never report an opaque load time.** Every swap is timed in four phases:
  evict, storage read, page-in, warm-up. If you add a code path that loads a
  model, it must produce a `SwapRecord` with all four.
- **Propagate `None`, never guess zero.** A counter that is unavailable on this
  kernel is `None` in the result row. A missing value must be visibly missing
  in analysis, not silently averaged in as zero.
- **Cold and warm are different measurements**, not noisy versions of each
  other. Every row records which one it was, and whether dropping the cache
  actually succeeded.
- **Score outside the timed region.** Quality scoring must never run between a
  start and stop timer.
- **Flush every result line immediately.** Runs take hours on a Pi and get
  killed; partial results must survive.
- **Keep prompts identical across arms.** A quality comparison confounded by
  prompt differences is worthless.
- Standard library plus the packages already in `requirements.txt`. Import
  `llama_cpp`, `datasets` and `matplotlib` lazily so the harness stays
  importable on a machine without them.
- Plain functions and small classes. No framework, no dependency injection, no
  abstract base classes. Prashanth should be able to read any module in one
  sitting.

## Working agreements

- Prashanth is a bootcamp student doing this part time over about 8 weeks.
  Prefer the smallest change that answers the current question. When there is a
  simple way and a general way, take the simple way and say what you skipped.
- **Explain the measurement reasoning, not just the code.** If a change affects
  what the numbers mean, say so explicitly before writing it.
- When a result looks surprising, suspect the harness before the hardware.
  The usual culprits, in order: a warm cache that should have been cold, a
  ceiling that was never enforced, a realised switch rate that drifted from the
  target, and thermal throttling.
- Do not add new dependencies without saying why the standard library will not
  do.
- Do not invent numbers. If something has not been measured yet, leave the
  placeholder and say it is unmeasured.

## Next steps, roughly in order

1. **Validate the harness before trusting it.** Confirm the realised switch rate
   tracks the target, that `run_in_cgroup.sh` really caps memory (try a ceiling
   below the model size and check it gets OOM-killed), and that cold and warm
   runs differ as expected.
2. **Get one full sweep on the laptop.** Both arms, all switch rates, warm
   cache. Produces the first `r*`, which is a sanity check rather than a result.
3. **Fine-tune the two specialists** on code and math, and verify the quality
   gap over the generalist is real. Without that gap the whole premise is dead,
   so check it early rather than in week 6.
4. **Add the Pi 5, microSD and NVMe.** This pair isolates the storage medium
   with everything else fixed, and is expected to be the largest single effect
   in the study.
5. **Cold-cache runs under a real ceiling,** 30 or more swaps per cell for
   stable medians and p95.
6. **Arm C, adapter swapping.** Expected to largely dissolve the trade-off, and
   that result alone would be worth the paper.
7. **Ablations:** ceiling sweep, quantization, prefetch, router accuracy,
   burstiness at fixed `r`.

## Paper

`../paper/main.tex` compiles with `pdflatex && bibtex && pdflatex && pdflatex`.
Its tables use `XX.X` placeholders whose rows match
`experiments/log_template.md` and the output of `src/analyze.py`. When asked to
update the paper with results, replace only the placeholder cells and keep the
table structure.
