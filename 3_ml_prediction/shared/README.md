# 3_ml_prediction/shared

Model-agnostic code shared by the per-model prediction drivers.

- `targets.py` — the benchmark targets (PDB id, kind, deposited chain
  sequences with `O` = hydroxyproline), loaded from the manifest via
  `data_locations`.
- `bench_common.py` — machinery for the multi-seed benchmark sweeps: the
  length-stratified target subset, the shared seed set, the randomised
  (target, seed) call plan, a manifest that every runner asserts against, a
  checkpointing CSV writer, and per-call CIF retention.
- `score_seed_sensitivity.py` — scores seed-sweep structures against the
  experimental references, using the same machinery as `4_scoring/score.py`.

## Seed-sensitivity scoring

The benchmark scores one diffusion sample from one seed per learned model.
To check whether those scores depend on the seed, each co-folder cost runner
(`4_scoring/compute_cost/bench_*.py`, entrypoint `bench_seeds`) can sweep
seeds 1–5 on a length-stratified subset and keep every structure it
generates. This directory scores those structures:

```bash
bash scripts/get_usalign.sh   # one time; scoring needs the US-align binary

python 3_ml_prediction/shared/score_seed_sensitivity.py
```

Inputs and outputs live in `4_scoring/compute_cost/results/` (gitignored),
alongside the timing records from the same runs.
