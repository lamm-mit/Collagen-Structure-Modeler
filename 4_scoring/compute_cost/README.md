# Compute-cost benchmark

Per-structure runtime and dollar cost for every method, measured on the same
targets. Dollars are derived from a dated price table so the numbers can be
recomputed when prices change.

## Files

- `prices.py` — Modal's dated price list (USD per second per resource) and
  the seconds → dollars map. This is the only file that introduces currency;
  everything upstream records seconds.
- `bench_cdsm.py` — times the deterministic CDSM pipeline locally,
  single-threaded (`OPENMM_CPU_THREADS=1`), via `run_pipeline.py --timings`.
  Records both the full 4-stage run and the headline stage alone.
- `bench_chai.py`, `bench_protenix.py`, `bench_boltz.py`, `bench_af3.py` —
  one Modal app per co-folding model. Images and inference settings match the
  production runners in `3_ml_prediction/`; only instrumentation (wall clock,
  NVML peak VRAM, cold/warm phase) is added.

The tier of record is L40S. Set `BENCH_GPU=L4` to re-measure on the cheaper
tier; records then write to `*_L4.csv` files so they cannot overwrite the
tier-of-record measurements.

## Running

```bash
# CDSM (local CPU, no GPU)
python 4_scoring/compute_cost/bench_cdsm.py --all

# Co-folding models (Modal GPU)
modal run 4_scoring/compute_cost/bench_chai.py::bench
modal run 4_scoring/compute_cost/bench_protenix.py::bench
modal run 4_scoring/compute_cost/bench_boltz.py::bench
modal run 4_scoring/compute_cost/bench_af3.py::bench
```

## Seed sweep

Each co-folder has a `bench_seeds` entrypoint that repeats the measurement
across seeds 1–5 (shared machinery in `3_ml_prediction/shared/bench_common.py`):
randomised (target, seed) call order, checkpointed after every call, and every
call's structure kept under `results/seed_cifs/` for the seed-sensitivity
scoring in `3_ml_prediction/shared/`.

```bash
modal run 4_scoring/compute_cost/bench_chai.py::bench_seeds
modal run 4_scoring/compute_cost/bench_protenix.py::bench_seeds
modal run 4_scoring/compute_cost/bench_boltz.py::bench_seeds
modal run 4_scoring/compute_cost/bench_af3.py::bench_seeds
BENCH_GPU=L4 modal run 4_scoring/compute_cost/bench_boltz.py::bench_seeds   # etc.
```

## Notes

- The local AF3 run is 1 seed / 1 diffusion sample / no MSA, for parity with
  the other co-folders. This is not the configuration of the scored af3
  predictions (AlphaFold Server, 5 samples, forced MSA) — keep the two apart
  when quoting both.
- AF3 model weights are mounted read-only from a Modal Volume and are never
  copied into the image, never logged, and never returned from the container;
  Google's Weights Terms of Use forbid redistribution. Running the AF3
  benchmark requires your own copy of the weights in the Volume.
- A call that produces no structure raises rather than recording a short,
  plausible-looking wall time — failed runs must not enter a cost median.
- AF3 runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the official images
  preallocate ~95% of the card, which would make measured peak VRAM
  meaningless).
- Modal bills GPU, CPU and memory together for a container's whole lifetime,
  including image builds and cold starts; the per-call records are
  in-container wall clock, i.e. a lower bound on billed time. Take total spend
  from `modal billing report`, never from summing per-call costs.

## Output locations

- Records and sweep structures: `4_scoring/compute_cost/results/`
  (gitignored).
- CDSM rebuilds: `4_scoring/compute_cost/rebuild/` (gitignored), never
  `2_deterministic_build/outputs/` — that path is the working-tree half of the
  `cdsm/*` dataset prefixes, so structures left there would shadow the
  published ones for any reader without `COLLAGEN_DATA_ROOT` set.
- Deliberately not `4_scoring/results/`: that is the working-tree half of the
  `scores` prefix, so creating it would shadow the HuggingFace copy of
  `scores_summary.csv`.
