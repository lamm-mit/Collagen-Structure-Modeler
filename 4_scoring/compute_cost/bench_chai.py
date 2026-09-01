#!/usr/bin/env python3
"""
bench_chai.py — measure Chai-1 compute cost on Modal.

Reuses the production image from 3_ml_prediction/chai/chai_modal_app.py
verbatim and keeps every inference knob identical (3 recycles / 200 diffusion
steps / 1 sample / seed 42 / ESM on / no MSA), so the cost measured here pairs
with the accuracy already scored.

What is measured, per call:
  wall_s        inference wall-clock inside the container
  peak_vram_mb  peak GPU memory, sampled via NVML on a background thread
  phase         cold (first call in a fresh container) or warm

Cost is derived in prices.py from Modal's dated price list; nothing here knows
about dollars. If the weights Volume is empty, the first call pays the full
cold start (~10 GB of weights incl. ESM2-3B); that is captured as a separate
cold row rather than smeared across targets.

Usage:
    modal run 4_scoring/compute_cost/bench_chai.py::bench
    modal run 4_scoring/compute_cost/bench_chai.py::bench_seeds
"""

import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, REPO, os.path.join(REPO, "3_ml_prediction", "shared"),
          os.path.join(REPO, "3_ml_prediction", "chai")):
    sys.path.insert(0, p)


MINUTES = 60
CACHE_DIR = "/cache"
GPU = os.environ.get("BENCH_GPU", "L40S")   # tier of record; BENCH_GPU=L4 to re-measure

app = modal.App("collagen-cost-bench-chai")

chai_volume = modal.Volume.from_name("collagen-chai1-weights", create_if_missing=True)

# Image identical to the production app, plus nvidia-ml-py for VRAM sampling.
chai_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chai_lab==0.6.1", "huggingface-hub", "nvidia-ml-py")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .env({"CHAI_DOWNLOADS_DIR": f"{CACHE_DIR}/chai"})
)

# Set on first call in a container; distinguishes cold from warm.
_CONTAINER_USED = False


class VramSampler:
    """Sample GPU memory on a background thread; report the peak."""

    def __init__(self, interval=0.1):
        self.interval, self.peak_mb, self._stop = interval, 0.0, False

    def __enter__(self):
        import threading

        import pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._pynvml = pynvml

        def loop():
            while not self._stop:
                used = pynvml.nvmlDeviceGetMemoryInfo(self._h).used / 1e6
                self.peak_mb = max(self.peak_mb, used)
                import time as _t
                _t.sleep(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join(timeout=2)
        try:
            self._pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _gpu_name():
    import pynvml
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    pynvml.nvmlShutdown()
    return n.decode() if isinstance(n, bytes) else n


@app.function(image=chai_image, gpu=GPU, volumes={CACHE_DIR: chai_volume},
              timeout=60 * MINUTES, max_containers=1)
def chai_fold_timed(name: str, fasta: str, use_msa: bool = False,
                    want_cif: bool = False, seed: int = 42) -> dict:
    """One Chai-1 fold, instrumented. Inference config matches chai_modal_app.py."""
    global _CONTAINER_USED
    import shutil
    import time
    from pathlib import Path

    import torch
    from chai_lab.chai1 import run_inference

    phase = "warm" if _CONTAINER_USED else "cold"
    _CONTAINER_USED = True

    workdir = Path("/tmp") / name
    out_dir = workdir / "out"
    shutil.rmtree(workdir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "input.fasta").write_text(fasta)

    torch.cuda.reset_peak_memory_stats()
    _t0_iso = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with VramSampler() as vs:
        run_inference(
            fasta_file=workdir / "input.fasta", output_dir=out_dir,
            num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=1,
            seed=seed, device=torch.device("cuda:0"),
            use_esm_embeddings=True, use_msa_server=use_msa,
        )
    wall = time.perf_counter() - t0
    chai_volume.commit()

    # No structure means no measurement.
    import glob
    if not glob.glob(str(out_dir / "*.cif")):
        raise RuntimeError(f"chai produced no CIF for {name} after {wall:.1f}s")

    rec = {"method": "chai", "pdb_id": name, "phase": phase,
           "t_start_utc": _t0_iso, "use_msa": use_msa,
           "seed": seed, "wall_s": round(wall, 3),
           "peak_vram_mb": round(vs.peak_mb, 1),
           "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
           "gpu": _gpu_name()}
    if want_cif:
        import glob as _g
        best, bs = None, -float("inf")
        for npz in _g.glob(str(out_dir / "scores.model_idx_*.npz")):
            import numpy as _np, re as _re
            try:
                sc = float(_np.asarray(_np.load(npz)["aggregate_score"]).mean())
            except Exception:  # noqa: BLE001
                sc = -float("inf")
            if sc > bs:
                bs, best = sc, int(_re.search(r"model_idx_(\d+)", npz).group(1))
        rec["cif"] = (out_dir / f"pred.model_idx_{best or 0}.cif").read_text()
    return rec


def _tier_suffix():
    """Non-default tiers write to their own file, so a comparison run
    cannot overwrite the tier-of-record measurements."""
    return "" if GPU == "L40S" else f"_{GPU}"


def _save(rows, name):
    import pandas as pd
    from data_locations import local_out
    out = local_out("4_scoring", "compute_cost", "results")
    path = os.path.join(out, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n-> {path}")
    return path


@app.local_entrypoint()
def bench(repeats: int = 2, n_targets: int = 10):
    from run_chai import build_fasta
    ts = _bench_targets(n_targets)
    print(f"Chai-1 on {GPU}: {len(ts)} targets x {repeats} repeats "
          f"({[sum(len(s) for s in t.chain_sequences()) for t in ts]} residues)")
    rows = []
    for rep in range(1, repeats + 1):
        for t in ts:
            r = chai_fold_timed.remote(t.pdb_id, build_fasta(t))
            r["repeat"] = rep
            r["n_residues"] = sum(len(s) for s in t.chain_sequences())
            rows.append(r)
            print(f"  rep{rep} {r['pdb_id']:6s} {r['phase']:5s} "
                  f"{r['wall_s']:8.2f}s  VRAM {r['peak_vram_mb']:8.0f} MB", flush=True)
    _save(rows, f"chai_records{_tier_suffix()}.csv")


# ── seed sweep ───────────────────────────────────────────────────────────────
# The published runs used one diffusion sample from one fixed seed, and the two
# "repeats" were repeated timings of that seed rather than independent draws.
# This entrypoint re-runs the same targets across BENCH_SEEDS. It writes only
# *_records_seeds.csv; the original bench() and its records are untouched.
@app.local_entrypoint()
def bench_seeds(seeds: str = "1,2,3,4,5", n_targets: int = 10,
                force: bool = False, shuffle: bool = True):
    from run_chai import build_fasta
    seeds = parse_seeds(seeds)
    ts = _bench_targets(n_targets)
    sync_manifest(ts, seeds, "chai")
    tag = "_shuffled" if shuffle else ""
    w = RecordWriter(f"chai_records_seeds{tag}{_tier_suffix()}.csv", force=force)
    plan = _bench_common().call_plan(ts, seeds, shuffle=shuffle)
    print(f"chai on {GPU}: {len(ts)} targets x {len(seeds)} seeds "
          f"= {len(plan)} calls ({'randomised' if shuffle else 'seed-blocked'} order)")

    for idx, (t, seed) in enumerate(plan):
        if w.skip(t.pdb_id, seed):
            print(f"  [{idx:2d}] seed{seed} {t.pdb_id:6s} already done, skipping",
                  flush=True)
            continue
        r = chai_fold_timed.remote(t.pdb_id, build_fasta(t),
                                   want_cif=True, seed=seed)
        save_cif("chai", seed, t.pdb_id, r.pop("cif", None), tier=_tier_suffix())
        r["call_index"] = idx          # lets drift be regressed out afterwards
        r["order"] = "shuffled" if shuffle else "seed_blocked"
        r["n_residues"] = sum(len(x) for x in t.chain_sequences())
        r["num_samples"] = 1
        w.add(r)
        print(f"  [{idx:2d}] seed{seed} {r['pdb_id']:6s} {r['phase']:5s} "
              f"{r['wall_s']:8.2f}s  VRAM {r['peak_vram_mb']:8.0f} MB", flush=True)
    print(f"\n-> {w.path}  ({len(w.rows)} calls)")


# bench_common lives in 3_ml_prediction/shared but is NOT mounted into the
# Modal container (only the entrypoint script is), so it must never be imported
# at module level — that would crash-loop every container. Local entrypoints
# run on the submitting machine, so importing inside them is safe.
def _bench_common():
    import bench_common
    return bench_common


def _bench_targets(n=10):
    return _bench_common().bench_targets(n)


def parse_seeds(spec):
    return _bench_common().parse_seeds(spec)


def sync_manifest(targets, seeds, model):
    return _bench_common().sync_manifest(targets, seeds, model)


def save_cif(model, seed, pdb_id, text, tier=""):
    return _bench_common().save_cif(model, seed, pdb_id, text, tier=tier)


def RecordWriter(filename, force=False):
    return _bench_common().RecordWriter(filename, force=force)
