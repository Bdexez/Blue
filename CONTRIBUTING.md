# Contributing

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch            # or the CPU wheel index used in CI
.venv/bin/python -m pip install -e '.[vision,dev]'
```

## Before committing

```bash
.venv/bin/ruff check blue tools tests
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -n auto -q tests   # ~2 min on 8+ cores; serial unittest takes >25 min
```

CUDA smoke tests in `tests/test_cuda_smoke.py` run only when a GPU is available. Keep large run outputs in the gitignored `runs/` directory: on many Linux systems `/tmp` is a RAM-backed tmpfs, and multi-gigabyte checkpoints there cause swapping and spurious failures of the multi-process tests.

## Changing numerical code

Refactors of the model must not change its outputs. Record reference outputs before the change and compare after it, on the same machine:

```bash
.venv/bin/python tools/golden_outputs.py record --output runs/golden-before.pt
# ... edit ...
.venv/bin/python tools/golden_outputs.py record --output runs/golden-after.pt
.venv/bin/python tools/golden_outputs.py compare runs/golden-before.pt runs/golden-after.pt
```

The recording covers logits, forecasts, losses, every parameter gradient and greedy, stochastic, speculative, Jacobi and beam generations (with left padding) for 14 feature combinations. A pure refactor should compare with tolerance 0; reordering floating-point operations typically stays below `--tolerance 1e-5`. `python -m blue verify` additionally checks causality, cache equivalence, padding and geometric invariances, gradient checkpointing and finite gradients for any configuration.

## Measuring performance

```bash
.venv/bin/python tools/benchmark_hotpaths.py --device cpu --device cuda --output runs/hotpaths.json
```

GPU timings are noisy: alternate the two versions and keep the best of at least two runs. Small presets are launch-bound on GPU, so an optimisation of a single kernel may not show end to end; report the end-to-end number (`train` writes `step_seconds` in `metrics.jsonl`).

GPU kernels are not bit-reproducible by default. Use the global `--deterministic` flag when comparing GPU training runs or resumes bit for bit.

## Reports

Technical reports (`AUDIT.md`, `IMPROVEMENTS.md`, `VALIDATION.md`, …) are written in French and record measured numbers with their conditions; the README is in English.
