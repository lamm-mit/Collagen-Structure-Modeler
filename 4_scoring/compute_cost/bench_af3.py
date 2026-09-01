#!/usr/bin/env python3
"""
bench_af3.py — measure AlphaFold 3 compute cost on Modal.

The scored predictions in `predictions/af3_msa/` came from AlphaFold Server (5
samples, forced MSA, templates off). This measures a *local* run at
single-seed, single-sample, no-MSA — matching how Boltz-2 / Chai-1 / Protenix
were run, and therefore comparable to their cost rows, but NOT the configuration
that produced the af3 accuracy numbers. Both facts must be carried into any
cost table built from these records; see README.md in this directory.

Running without MSA also avoids AF3's ~630 GB genetic databases entirely:
`--norun_data_pipeline` skips the search, and the input JSON supplies empty
MSAs and no templates.

Model parameters are mounted read-only from the `af3-models` Volume (expected
there as af3_parameters.bin.zst). They are never copied into the image, never
logged, and never leave the container — Google's Weights Terms of Use forbid
redistribution.

Usage:
    modal run 4_scoring/compute_cost/bench_af3.py::smoke     # one target first
    modal run 4_scoring/compute_cost/bench_af3.py::bench
    modal run 4_scoring/compute_cost/bench_af3.py::bench_seeds
"""

import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, REPO, os.path.join(REPO, "3_ml_prediction", "shared"),
          os.path.join(REPO, "3_ml_prediction", "af3")):
    sys.path.insert(0, p)


MINUTES = 60
MODEL_DIR = "/models"          # af3-models Volume, holds af3_parameters.bin.zst
CACHE_DIR = "/cache"           # scratch for the (unused) database dir
GPU = os.environ.get("BENCH_GPU", "L40S")
# Image build stays pinned: it is content-addressed, and varying it
# would force a full CUDA rebuild for every tier measured.
BUILD_GPU = "L40S"

app = modal.App("collagen-af3-cost")

af3_models = modal.Volume.from_name("af3-models")
af3_scratch = modal.Volume.from_name("collagen-af3-scratch", create_if_missing=True)

# Mirrors the official docker/Dockerfile: CUDA 12.6 + Python 3.12, dependencies
# resolved by `uv sync --frozen` against the repo's uv.lock, then `build_data` to
# construct the chemical-components database. The repo has no requirements.txt —
# it is uv/pyproject-based.
#
# HMMER is deliberately omitted: it serves only the genetic-search data
# pipeline, which --norun_data_pipeline skips. That also avoids AF3's ~630 GB of
# databases entirely.
#
# L40S is Ada (sm_89), above AF3's stated Compute Capability 8.0 minimum.
AF3_DIR = "/opt/alphafold3"
AF3_TAG = "v3.0.4"        # pinned release; see the clone comment below
AF3_VENV = f"{AF3_DIR}/.venv"    # created by `uv sync`
af3_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04",
                              add_python="3.12")
    .apt_install("git", "wget", "gcc", "g++", "make", "cmake", "zlib1g-dev", "zstd")
    # CC/CXX must be visible to every build step, not just one: uv compiles
    # AF3's C++ extensions in an isolated environment where CMake otherwise
    # fails with "CMAKE_CXX_COMPILER not set, after EnableLanguage". Set at
    # image level because each run_commands entry is its own shell, so an
    # `export` in one does not reach the next (build_data rebuilds too).
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .run_commands(
        "pip install uv",
        # Clone a *tagged release*, with full history. Two reasons, both learnt
        # the hard way: a --depth 1 clone breaks the git-derived version in the
        # build backend, and cloning main puts HEAD past the last tag, so the
        # rebuilt version ("3.0.5.dev3+g29596b970") no longer matches uv.lock
        # and scikit-build-core aborts with "Metadata mismatch in METADATA".
        f"git clone --branch {AF3_TAG} "
        f"https://github.com/google-deepmind/alphafold3.git {AF3_DIR}",
        f"cd {AF3_DIR} && uv sync --frozen --all-groups --no-editable",
        # NOT `uv run build_data`: uv re-syncs on every `uv run`, which rebuilds
        # the project, and that rebuild fails with "Metadata mismatch in
        # METADATA" from scikit-build-core. `uv sync` above already installed
        # everything, so invoke the venv's own entry point and skip the rebuild.
        f"{AF3_VENV}/bin/build_data",
        gpu=BUILD_GPU,    # jax/CUDA wheels resolve against a visible GPU
    )
    # Into Modal's own Python (the function body's interpreter), not AF3's uv
    # venv — the venv is only used by the run_alphafold.py subprocess.
    .pip_install("nvidia-ml-py")
    .env({
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",   # official setting
        # Official images set PREALLOCATE=true with MEM_FRACTION=0.95, which
        # grabs ~95% of the card up front and would make a measured peak-VRAM
        # figure meaningless (NVML would just report the reservation). Turned
        # off so peak memory is observable and the minimum-viable-tier analysis
        # is possible. Deliberate deviation, noted in README.md.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
)


class VramSampler:
    """Peak GPU memory via NVML, sampled on a background thread."""

    def __init__(self, interval=0.25):
        self.interval, self.peak_mb, self._stop = interval, 0.0, False

    def __enter__(self):
        import threading
        import time as _t

        import pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._pynvml = pynvml

        def loop():
            while not self._stop:
                self.peak_mb = max(
                    self.peak_mb,
                    pynvml.nvmlDeviceGetMemoryInfo(self._h).used / 1e6)
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
    """The GPU actually in use, rather than the tier we asked for."""
    import pynvml
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    pynvml.nvmlShutdown()
    return n.decode() if isinstance(n, bytes) else n


_CONTAINER_USED = False


@app.function(image=af3_image, gpu=GPU,
              volumes={MODEL_DIR: af3_models, CACHE_DIR: af3_scratch},
              timeout=90 * MINUTES, max_containers=1)
def af3_fold_timed(name: str, job: dict, want_cif: bool = False,
                   num_samples: int = 1, seed: int = 1) -> dict:
    """One AlphaFold 3 prediction, instrumented. Returns a timing record."""
    global _CONTAINER_USED
    import glob
    import subprocess
    import time
    from pathlib import Path

    phase = "warm" if _CONTAINER_USED else "cold"
    _CONTAINER_USED = True

    # AF3 expects the parameters as <model_dir>/af3.bin[.zst]. The Volume holds
    # them under their original name, and is mounted read-only in practice, so
    # expose a correctly-named symlink in a writable scratch dir. No copy, no
    # read of the weights themselves.
    params_dir = Path("/tmp/af3_params")
    params_dir.mkdir(parents=True, exist_ok=True)
    link = params_dir / "af3.bin.zst"
    if not link.exists():
        link.symlink_to(f"{MODEL_DIR}/af3_parameters.bin.zst")

    # Unique per (name, seed), NOT just per name. AF3 redirects to a
    # timestamped sibling when the output dir is non-empty, so a shared
    # /tmp/<name>/out meant calls 2..5 of the seed sweep ran real inference
    # into out_<ts>/ while the glob below re-harvested call 1's CIF — every
    # seed "prediction" was byte-identical to seed 1's. --force_output_dir
    # additionally makes a repeated (name, seed) call overwrite in place
    # instead of redirecting.
    work = Path("/tmp") / f"{name}_s{seed}"
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = work / "input.json"
    in_path.write_text(json.dumps(job))

    # AF3's dependencies live in the repo's uv venv. Its interpreter is called
    # directly rather than through `uv run`, which would re-sync and rebuild
    # (see the image comment on build_data).
    cmd = [
        f"{AF3_VENV}/bin/python", "run_alphafold.py",
        f"--json_path={in_path}",
        f"--model_dir={params_dir}",
        f"--output_dir={out_dir}",
        "--force_output_dir",             # overwrite, never redirect to out_<ts>
        "--norun_data_pipeline",          # no MSA search; no 630 GB databases
        # ALWAYS explicit. The flag defaults to 5, so omitting it silently runs
        # 5x the diffusion work of the 1-sample Boltz/Chai/Protenix runs.
        f"--num_diffusion_samples={num_samples}",
    ]
    print(f"[{name}] {' '.join(cmd)}", flush=True)
    _t0_iso = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with VramSampler() as vs:
        proc = subprocess.run(cmd, cwd=AF3_DIR, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    cifs = sorted(glob.glob(str(out_dir / "**" / "*.cif"), recursive=True))
    if not cifs:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        raise RuntimeError(f"no CIF produced (rc={proc.returncode})\n{tail}")
    best = next((c for c in cifs if "model.cif" in c or "_model_0" in c), cifs[0])

    rec = {"method": "af3", "pdb_id": name, "phase": phase,
           "t_start_utc": _t0_iso, "seed": seed,
           "wall_s": round(wall, 3), "peak_vram_mb": round(vs.peak_mb, 1),
           "torch_peak_mb": None, "rc": proc.returncode, "gpu": _gpu_name()}
    if want_cif:
        rec["cif"] = Path(best).read_text()
    return rec


# ── input encoding ───────────────────────────────────────────────────────────
def build_af3_job(target, seed: int = 1) -> dict:
    """AF3 (local dialect) job: HYP as a CCD modification on a Pro parent.

    Mirrors make_af_server_json.py, with the two differences the local dialect
    requires: ptmType is the bare CCD code (not CCD_-prefixed), and empty
    unpairedMsa/pairedMsa/templates are supplied explicitly so the data pipeline
    can be skipped.
    """
    def chain(seq, ids):
        base = "".join("P" if c == "O" else c for c in seq)
        mods = [{"ptmType": "HYP", "ptmPosition": i + 1}
                for i, c in enumerate(seq) if c == "O"]
        p = {"id": ids, "sequence": base,
             "unpairedMsa": "", "pairedMsa": "", "templates": []}
        if mods:
            p["modifications"] = mods
        return {"protein": p}

    if target.is_homotrimer:
        sequences = [chain(target.sequences[0], ["A", "B", "C"])]
    else:
        sequences = [chain(s, [c]) for s, c in zip(target.sequences, "ABC")]
    return {"name": target.pdb_id, "modelSeeds": [seed], "sequences": sequences,
            "dialect": "alphafold3", "version": 1}


@app.local_entrypoint()
def smoke():
    """One target, to shake out the image and the input encoding cheaply."""
    t = _targets(10)[0]
    r = af3_fold_timed.remote(t.pdb_id, build_af3_job(t))
    print(f"\n  {r['pdb_id']} {r['phase']} {r['wall_s']:.1f}s  "
          f"VRAM {r['peak_vram_mb']:.0f} MB  rc={r['rc']}")


@app.local_entrypoint()
def bench(repeats: int = 2, n_targets: int = 10, num_samples: int = 1):
    """Cost benchmark. num_samples=1 matches the Boltz/Chai/Protenix runs;
    num_samples=5 matches AlphaFold Server. Written to a per-arm CSV."""
    import pandas as pd

    from data_locations import local_out

    ts = _targets(n_targets)
    print(f"AlphaFold 3 on {GPU}: {len(ts)} targets x {repeats} repeats "
          f"(1 seed, {num_samples} diffusion sample(s), no MSA, no templates)")
    rows = []
    for rep in range(1, repeats + 1):
        for t in ts:
            r = af3_fold_timed.remote(t.pdb_id, build_af3_job(t),
                                      num_samples=num_samples)
            r["repeat"] = rep
            r["num_samples"] = num_samples
            r["n_residues"] = sum(len(s) for s in t.chain_sequences())
            rows.append(r)
            print(f"  rep{rep} {r['pdb_id']:6s} {r['phase']:5s} "
                  f"{r['wall_s']:8.2f}s  VRAM {r['peak_vram_mb']:8.0f} MB", flush=True)
    out = local_out("4_scoring", "compute_cost", "results")
    suffix = "" if num_samples == 1 else f"_s{num_samples}"
    suffix += "" if GPU == "L40S" else f"_{GPU}"
    path = os.path.join(out, f"af3_records{suffix}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n-> {path}")


# ── seed sweep ───────────────────────────────────────────────────────────────
# Same targets as bench(), across BENCH_SEEDS instead of repeated timings of one
# seed. num_samples stays 1 so per-call cost remains comparable to the other
# models — this varies the seed only, not the diffusion sample count.
# Writes only af3_records_seeds.csv; bench() and its records are untouched.
@app.local_entrypoint()
def bench_seeds(seeds: str = "1,2,3,4,5", n_targets: int = 10,
                force: bool = False, shuffle: bool = True):
    seeds = parse_seeds(seeds)
    ts = _targets(n_targets)
    sync_manifest(ts, seeds, "af3")
    tag = "_shuffled" if shuffle else ""
    w = RecordWriter(f"af3_records_seeds{tag}{'' if GPU == 'L40S' else f'_{GPU}'}.csv", force=force)
    plan = _bench_common().call_plan(ts, seeds, shuffle=shuffle)
    print(f"af3 on {GPU}: {len(ts)} targets x {len(seeds)} seeds "
          f"= {len(plan)} calls ({'randomised' if shuffle else 'seed-blocked'} order)")

    for idx, (t, seed) in enumerate(plan):
        if w.skip(t.pdb_id, seed):
            print(f"  [{idx:2d}] seed{seed} {t.pdb_id:6s} already done, skipping",
                  flush=True)
            continue
        r = af3_fold_timed.remote(t.pdb_id, build_af3_job(t, seed), want_cif=True,
                                  num_samples=1, seed=seed)
        save_cif("af3", seed, t.pdb_id, r.pop("cif", None), tier=('' if GPU == 'L40S' else f'_{GPU}'))
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


def _targets(n=10):
    return _bench_common().bench_targets(n)


def parse_seeds(spec):
    return _bench_common().parse_seeds(spec)


def sync_manifest(targets, seeds, model):
    return _bench_common().sync_manifest(targets, seeds, model)


def save_cif(model, seed, pdb_id, text, tier=""):
    return _bench_common().save_cif(model, seed, pdb_id, text, tier=tier)


def RecordWriter(filename, force=False):
    return _bench_common().RecordWriter(filename, force=force)
