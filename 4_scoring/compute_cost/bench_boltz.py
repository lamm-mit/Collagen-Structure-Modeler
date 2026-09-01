#!/usr/bin/env python3
"""
bench_boltz.py — measure self-hosted Boltz-2 compute cost on Modal.

The benchmark's Boltz-2 predictions were bought from the hosted `boltz-api`
at a per-prediction price — a price, not a measurement, on unknown hardware.
This runs the open weights on the same GPU tier the other co-folders are
measured on, which does two things: it puts Boltz-2 on the same footing as
Chai-1 and Protenix, and it shows what the hosted API's price actually buys.

Configuration matches run_boltz.py: empty MSA (single-sequence), one diffusion
sample, HYP as a CCD modification on a Pro parent. Boltz's own weights (~2 GB)
are cached in a Modal Volume so only the first call pays for them.

Usage:
    modal run 4_scoring/compute_cost/bench_boltz.py::smoke
    modal run 4_scoring/compute_cost/bench_boltz.py::bench
    modal run 4_scoring/compute_cost/bench_boltz.py::bench_seeds
"""

import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, REPO, os.path.join(REPO, "3_ml_prediction", "shared"),
          os.path.join(REPO, "3_ml_prediction", "boltz")):
    sys.path.insert(0, p)


MINUTES = 60
CACHE_DIR = "/cache"
GPU = os.environ.get("BENCH_GPU", "L40S")

app = modal.App("collagen-boltz-cost")
boltz_volume = modal.Volume.from_name("collagen-boltz-weights", create_if_missing=True)

boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install("boltz[cuda]", "nvidia-ml-py")
    .env({"BOLTZ_CACHE": f"{CACHE_DIR}/boltz"})
)


class VramSampler:
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


@app.function(image=boltz_image, gpu=GPU, volumes={CACHE_DIR: boltz_volume},
              timeout=90 * MINUTES, max_containers=1)
def boltz_fold_timed(name: str, yaml_text: str, want_cif: bool = False,
                     use_msa: bool = False, seed: int = 42) -> dict:
    """One Boltz-2 prediction from a YAML spec, instrumented."""
    global _CONTAINER_USED
    import glob
    import subprocess
    import time
    from pathlib import Path

    phase = "warm" if _CONTAINER_USED else "cold"
    _CONTAINER_USED = True

    work = Path("/tmp") / name
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = work / f"{name}.yaml"
    in_path.write_text(yaml_text)

    cmd = ["boltz", "predict", str(in_path),
           "--out_dir", str(out_dir),
           "--cache", f"{CACHE_DIR}/boltz",
           "--accelerator", "gpu", "--devices", "1",
           "--diffusion_samples", "1",
           "--seed", str(seed),
           "--output_format", "mmcif",
           "--override"]
    if use_msa:
        # Queries the ColabFold MMseqs2 service. Off by default: the benchmark
        # configuration is single-sequence (empty MSA).
        cmd.append("--use_msa_server")
    print(f"[{name}] {' '.join(cmd)}", flush=True)
    _t0_iso = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with VramSampler() as vs:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    boltz_volume.commit()

    cifs = sorted(glob.glob(str(out_dir / "**" / "*.cif"), recursive=True))
    if not cifs:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        raise RuntimeError(f"no CIF produced (rc={proc.returncode})\n{tail}")
    best = next((c for c in cifs if "model_0" in c), cifs[0])

    rec = {"method": "boltz", "pdb_id": name, "phase": phase,
           "t_start_utc": _t0_iso, "use_msa": use_msa,
           "seed": seed,
           "wall_s": round(wall, 3), "peak_vram_mb": round(vs.peak_mb, 1),
           "torch_peak_mb": None, "rc": proc.returncode, "gpu": _gpu_name()}
    if want_cif:
        rec["cif"] = Path(best).read_text()
    return rec


# ── input encoding ───────────────────────────────────────────────────────────
def build_boltz_yaml(target, use_msa: bool = False) -> str:
    """Boltz-2 YAML for a target. Mirrors run_boltz.py's API encoding:
    'P' at each HYP site plus a CCD modification, and an empty MSA.

    The local package takes YAML with 1-indexed `position`, where the hosted
    API took JSON with a 0-indexed `residue_index` — the one encoding
    difference between the two paths.
    """
    def block(seq, ids):
        base = "".join("P" if c == "O" else c for c in seq)
        lines = [f"  - protein:",
                 f"      id: [{', '.join(ids)}]",
                 f"      sequence: {base}"]
        if not use_msa:
            lines.append("      msa: empty")
        mods = [i + 1 for i, c in enumerate(seq) if c == "O"]
        if mods:
            lines.append("      modifications:")
            for pos in mods:
                lines += [f"        - position: {pos}", f"          ccd: HYP"]
        return "\n".join(lines)

    if target.is_homotrimer:
        blocks = [block(target.sequences[0], ["A", "B", "C"])]
    else:
        blocks = [block(s, [c]) for s, c in zip(target.sequences, "ABC")]
    return "version: 1\nsequences:\n" + "\n".join(blocks) + "\n"


@app.local_entrypoint()
def smoke():
    t = _targets(10)[0]
    print(build_boltz_yaml(t))
    r = boltz_fold_timed.remote(t.pdb_id, build_boltz_yaml(t))
    print(f"\n  {r['pdb_id']} {r['phase']} {r['wall_s']:.1f}s  "
          f"VRAM {r['peak_vram_mb']:.0f} MB  rc={r['rc']}")


def _tier(g):
    return "" if g == "L40S" else f"_{g}"


@app.local_entrypoint()
def bench(repeats: int = 2, n_targets: int = 10):
    import pandas as pd

    from data_locations import local_out

    ts = _targets(n_targets)
    print(f"Boltz-2 (self-hosted) on {GPU}: {len(ts)} targets x {repeats} repeats")
    rows = []
    for rep in range(1, repeats + 1):
        for t in ts:
            r = boltz_fold_timed.remote(t.pdb_id, build_boltz_yaml(t))
            r["repeat"] = rep
            r["n_residues"] = sum(len(s) for s in t.chain_sequences())
            rows.append(r)
            print(f"  rep{rep} {r['pdb_id']:6s} {r['phase']:5s} "
                  f"{r['wall_s']:8.2f}s  VRAM {r['peak_vram_mb']:8.0f} MB", flush=True)
    out = local_out("4_scoring", "compute_cost", "results")
    suffix = "" if GPU == "L40S" else f"_{GPU}"
    path = os.path.join(out, f"boltz_records{suffix}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n-> {path}")


# ── seed sweep ───────────────────────────────────────────────────────────────
# Same targets as bench(), across BENCH_SEEDS instead of repeated timings of one
# seed. Writes only boltz_records_seeds.csv; bench() and the records it produced
# are untouched.
@app.local_entrypoint()
def bench_seeds(seeds: str = "1,2,3,4,5", n_targets: int = 10,
                force: bool = False, shuffle: bool = True):
    seeds = parse_seeds(seeds)
    ts = _targets(n_targets)
    sync_manifest(ts, seeds, "boltz")
    tag = "_shuffled" if shuffle else ""
    w = RecordWriter(f"boltz_records_seeds{tag}{_tier(GPU)}.csv", force=force)
    plan = _bench_common().call_plan(ts, seeds, shuffle=shuffle)
    print(f"boltz on {GPU}: {len(ts)} targets x {len(seeds)} seeds "
          f"= {len(plan)} calls ({'randomised' if shuffle else 'seed-blocked'} order)")

    for idx, (t, seed) in enumerate(plan):
        if w.skip(t.pdb_id, seed):
            print(f"  [{idx:2d}] seed{seed} {t.pdb_id:6s} already done, skipping",
                  flush=True)
            continue
        r = boltz_fold_timed.remote(t.pdb_id, build_boltz_yaml(t), want_cif=True, seed=seed)
        save_cif("boltz", seed, t.pdb_id, r.pop("cif", None), tier=_tier(GPU))
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
