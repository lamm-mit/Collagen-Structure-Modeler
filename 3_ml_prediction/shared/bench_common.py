#!/usr/bin/env python3
"""
bench_common.py — shared target/seed definitions for the multi-seed sweep.

The target list was previously copy-pasted verbatim into the per-model
benchmark runners (the bench_*.py scripts in 4_scoring/compute_cost/).
Separate copies meant "all four models ran the same sequences" was true only
by coincidence of the copies agreeing; here it
is one definition plus a manifest the runners assert against.

The seed sweep answers a review comment: every published co-folder run used one
diffusion sample from one fixed seed, and the two "repeats" were repeated
timings of that single seed rather than independent draws.

Write safety: everything here writes only to *_seeds.csv / seed_cifs/ paths
under 4_scoring/compute_cost/results/. The pre-existing records are never
opened for writing — RecordWriter enforces that with an assertion on the
filename.
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, HERE):     # data_locations at the root, targets beside us
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Seeds are shared across all four models so the arms are directly comparable.
# Seed 1 reproduces the original Protenix and AF3 runs (both used seed 1), which
# doubles as a reproducibility check. The original Chai/Boltz runs used seed 42,
# so old and new records are NOT poolable for those two.
BENCH_SEEDS = (1, 2, 3, 4, 5)

MANIFEST = "bench_manifest.json"


def results_dir():
    """Sweep output lives with the compute-cost benchmark that generates it."""
    from data_locations import local_out
    return local_out("4_scoring", "compute_cost", "results")


def bench_targets(n=10):
    """Exactly n targets, stratified across the length range (48-108 residues).

    Guards two edge cases the naive form gets wrong: n=1 divides by zero, and
    rounding collisions silently yield fewer than n targets, so the reported
    sample size would disagree with the one requested.
    """
    from targets import load_targets
    ts = sorted(load_targets(), key=lambda t: sum(len(s) for s in t.chain_sequences()))
    n = max(1, min(n, len(ts)))
    if n == 1:
        return [ts[len(ts) // 2]]                    # median-length target
    picked = []
    for i in range(n):
        j = round(i * (len(ts) - 1) / (n - 1))
        while j < len(ts) and j in picked:           # collision: take next free
            j += 1
        if j >= len(ts):
            j = next(k for k in range(len(ts)) if k not in picked)
        picked.append(j)
    return [ts[j] for j in sorted(picked)]


def parse_seeds(spec):
    """'1,2,3' -> (1, 2, 3). Accepts an int or an iterable unchanged."""
    if spec is None:
        return tuple(BENCH_SEEDS)
    if isinstance(spec, int):
        return (spec,)
    if not isinstance(spec, str):
        return tuple(int(s) for s in spec)
    seeds = tuple(int(x) for x in str(spec).replace(" ", "").split(",") if x)
    if not seeds:
        raise ValueError(f"no seeds parsed from {spec!r}")
    return seeds


SHUFFLE_KEY = 20260822          # fixed, so the call order is reproducible


def call_plan(targets, seeds, shuffle=True):
    """The (target, seed) calls to make, in execution order.

    A seed-outer loop runs all targets at seed 1, then all at seed 2, and so on
    -- which makes seed index almost perfectly collinear with call order. The
    first sweep showed this: after removing target size, the residual wall time
    correlated with call order (r = +0.66 af3, +0.68 boltz, -0.37 chai,
    -0.59 protenix) just as strongly as with seed, and the drift ran in
    OPPOSITE directions per model. That is container warm-up and thermal
    behaviour, not the noise draw -- but the design cannot separate them.

    Interleaving (target-outer, seed-inner) balances seed against time; the
    shuffle additionally breaks any residual position effect. Keyed on
    SHUFFLE_KEY so the order is fixed and reproducible.
    """
    import random
    plan = [(t, s) for t in targets for s in seeds]     # target-outer, seed-inner
    if shuffle:
        random.Random(SHUFFLE_KEY).shuffle(plan)
    return plan


def _fingerprint(targets):
    """The identity every model must agree on: ids, sequences, residue counts."""
    return [{"pdb_id": t.pdb_id,
             "sequences": list(t.chain_sequences()),
             "n_residues": sum(len(s) for s in t.chain_sequences())}
            for t in targets]


def sync_manifest(targets, seeds, model):
    """Write the manifest on first use; afterwards assert this model matches it.

    This is what makes "the same sequences for every model prediction" an
    enforced invariant rather than an assumption about bench_targets() being
    deterministic across four separate invocations.

    Keyed by target count so a 1-target smoke run cannot poison the canonical
    10-target entry.
    """
    path = os.path.join(results_dir(), MANIFEST)
    key = str(len(targets))
    fp = _fingerprint(targets)
    seeds = list(seeds)

    man = {}
    if os.path.exists(path):
        with open(path) as fh:
            man = json.load(fh)

    if key not in man:
        man[key] = {"seeds": seeds, "targets": fp, "models_run": [model]}
        with open(path, "w") as fh:
            json.dump(man, fh, indent=2)
        print(f"  manifest[{key}] written: {len(fp)} targets, seeds {seeds}")
        return

    entry = man[key]
    if entry["targets"] != fp:
        old_ids = [t["pdb_id"] for t in entry["targets"]]
        new_ids = [t["pdb_id"] for t in fp]
        raise SystemExit(
            f"target set does not match {MANIFEST}[{key}].\n"
            f"  manifest: {old_ids}\n  this run: {new_ids}\n"
            "All models must benchmark identical sequences; refusing to run.")
    if not set(seeds) <= set(entry["seeds"]):
        raise SystemExit(f"seeds {seeds} not covered by manifest {entry['seeds']}; "
                         "refusing to run.")
    if model not in entry["models_run"]:
        entry["models_run"].append(model)
        with open(path, "w") as fh:
            json.dump(man, fh, indent=2)
    print(f"  manifest[{key}] OK: {len(fp)} targets, seeds {seeds} "
          f"(models so far: {', '.join(entry['models_run'])})")


class RecordWriter:
    """Append-only, resumable CSV writer for the seed sweep.

    Checkpoints after every call: 50 serial calls per model is long enough that
    a mid-run failure is a realistic outcome, and losing 40 completed GPU calls
    to a crash on the 41st is expensive. On resume, (pdb_id, seed) pairs already
    in the file are skipped.
    """

    def __init__(self, filename, force=False):
        # Hard guard: the pre-existing *_records.csv are never rewritten.
        if "_records_seeds" not in filename or not filename.endswith(".csv"):
            raise ValueError(
                f"RecordWriter refuses {filename!r}: this sweep may only write "
                "*_records_seeds*.csv files, never the existing records.")
        self.path = os.path.join(results_dir(), filename)
        self.rows = []
        self.done = set()
        if os.path.exists(self.path):
            if force:
                raise SystemExit(
                    f"{self.path} exists. Delete it by hand if you really mean "
                    "to discard those runs; --force will not overwrite it.")
            with open(self.path) as fh:
                for r in csv.DictReader(fh):
                    self.rows.append(r)
                    self.done.add((r["pdb_id"], int(r["seed"])))
            print(f"  resuming: {len(self.done)} calls already in {self.path}")

    def skip(self, pdb_id, seed):
        return (pdb_id, int(seed)) in self.done

    def add(self, rec):
        self.rows.append(rec)
        self.done.add((rec["pdb_id"], int(rec["seed"])))
        self.flush()

    def flush(self):
        if not self.rows:
            return
        cols = []
        for r in self.rows:                       # union, first-seen order
            for k in r:
                if k not in cols:
                    cols.append(k)
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        os.replace(tmp, self.path)                # atomic; never a partial file


def save_cif(model, seed, pdb_id, text, tier=""):
    """Keep the structure from every timed call.

    The runs happen either way and the CIFs are otherwise discarded; keeping
    them costs nothing and means the seed-variance question can be answered on
    these targets later without paying for the GPU time again.
    """
    if not text:
        return None
    d = os.path.join(results_dir(), "seed_cifs", f"{model}{tier}", f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{pdb_id}_{model}.cif")
    with open(path, "w") as fh:
        fh.write(text)
    return path
