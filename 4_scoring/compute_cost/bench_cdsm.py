#!/usr/bin/env python3
"""
bench_cdsm.py — measure the deterministic (CDSM) pipeline's compute cost.

Runs 2_deterministic_build/run_pipeline.py with per-step timing enabled and
converts the result into the benchmark's flat record schema.

Two facts, established by the determinism pre-check (see README.md in this
directory), fix the configuration:

  * OPENMM_CPU_THREADS=1 is the only *reproducible* setting. OpenMM's CPU
    platform is multi-threaded by default and varies nonbonded summation order
    run to run, so repeated builds of the same target differ by ~0.05 A. Pinning
    to one thread makes the pipeline bit-reproducible, and is also what makes
    single-core seconds honest.
  * The pipeline emits four stages, but only fullseq_reregistered_relaxed is
    used in the cross-method accuracy comparison. Both totals are reported: the
    full four-stage run, and the headline stage alone (steps 1-3 + one tleap +
    minimisation), which is what producing one scored structure actually costs.

Usage:
    python 4_scoring/compute_cost/bench_cdsm.py --all
    python 4_scoring/compute_cost/bench_cdsm.py --from-timings run.csv
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from data_locations import local_out  # noqa: E402
from prices import cost_usd  # noqa: E402

# Steps that make up the headline stage: the CDSM geometry, the one tleap call
# that builds its side chains, and the minimisation. The other three tleap calls
# build the core/extended/fullseq stages, which are diagnostic intermediates.
HEADLINE_STEPS = ["step1_backbone", "step2_extend_termini", "step3_reregister",
                  "step4_tleap_full_reregistered", "step5_minimize"]
GEOMETRY_STEPS = ["step1_backbone", "step2_extend_termini", "step3_reregister"]


def machine() -> dict:
    """Identify the CPU this was measured on — the claim is hardware-specific."""
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 — non-macOS or key absent
            return None

    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "model": sysctl("hw.model") or "unknown",
        "n_cores_total": sysctl("hw.ncpu"),
        "n_cores_perf": sysctl("hw.perflevel0.physicalcpu"),
        "n_cores_eff": sysctl("hw.perflevel1.physicalcpu"),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


# Benchmark rebuilds go here, NOT 2_deterministic_build/outputs/. That path is
# the working-tree half of the cdsm/* prefixes in data_locations.LAYOUT, so
# structures left there would (a) shadow the published ones for any reader
# without COLLAGEN_DATA_ROOT set, and (b) be pushed over them by
# `upload_to_huggingface.py --all`. This directory is in neither LAYOUT nor git.
REBUILD_ROOT = os.path.join(HERE, "rebuild")


def run_pipeline(timings_csv: str, ids_file: str = None,
                 output_root: str = REBUILD_ROOT) -> None:
    """Invoke the build with timing enabled and OPENMM_CPU_THREADS pinned."""
    env = dict(os.environ, OPENMM_CPU_THREADS="1")
    cmd = [sys.executable, "run_pipeline.py",
           "--timings", os.path.abspath(timings_csv),
           "--output-root", os.path.abspath(output_root)]
    cmd += ["--list", os.path.abspath(ids_file)] if ids_file else ["--all"]
    print(f"$ OPENMM_CPU_THREADS=1 {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=os.path.join(REPO, "2_deterministic_build"),
                   env=env, check=True)


def to_records(timings_csv: str, mach: dict) -> pd.DataFrame:
    """Convert per-step timings into the benchmark's flat record schema."""
    t = pd.read_csv(timings_csv).fillna(0.0)
    built = t[t.status == "ok"].copy()
    if built.empty:
        raise SystemExit(f"{timings_csv}: no rows with status == 'ok'")

    # A timings CSV from `--relax-method anneal` has step5_anneal, not
    # step5_minimize, and a run where every target hit KNOWN_ZERODIV never
    # emits the later columns at all. Say so rather than raising a bare KeyError.
    missing = [c for c in HEADLINE_STEPS if c not in built.columns]
    if missing:
        raise SystemExit(
            f"{timings_csv}: missing step column(s) {missing}.\n"
            f"  present: {sorted(c for c in built.columns if c.startswith('step'))}\n"
            f"  (a --relax-method anneal run records step5_anneal instead of "
            f"step5_minimize; this script costs the minimize pipeline.)")

    built["headline_s"] = built[HEADLINE_STEPS].sum(axis=1)
    built["geometry_s"] = built[GEOMETRY_STEPS].sum(axis=1)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recs = []
    for _, r in built.iterrows():
        for basis, wall in (("full_pipeline", r.total_s), ("headline_stage", r.headline_s)):
            recs.append({
                "run_id": f"cdsm-{basis}-{r.pdb_id}",
                "method": "fullseq_reregistered_relaxed",
                "pdb_id": r.pdb_id,
                "n_residues": int(r.n_residues),
                "tier": "cpu:1core",
                "phase": "warm",
                "repeat": 1,
                "basis": basis,
                "wall_s": round(wall, 4),
                "billed_basis": "local wall clock; costed at Modal 1-core rate",
                "peak_vram_mb": 0,               # no accelerator is used at all
                "cost_usd": cost_usd(wall, "cpu:core", n=1),
                "timestamp": now,
                "notes": f"{mach['cpu']}; OPENMM_CPU_THREADS=1; single-threaded",
            })
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="build and time every manifest ID")
    g.add_argument("--list", help="file with one PDB ID per line")
    g.add_argument("--from-timings", help="skip the build; reuse this timings CSV")
    ap.add_argument("--output-root", default=REBUILD_ROOT,
                    help=f"where the rebuilt stages go (default: {REBUILD_ROOT}). "
                         "Kept out of 2_deterministic_build/outputs so it cannot "
                         "shadow or overwrite the published structures.")
    args = ap.parse_args()

    out = local_out("4_scoring", "compute_cost", "results")
    timings_csv = args.from_timings or os.path.join(out, "cdsm_timings.csv")
    if not args.from_timings:
        run_pipeline(timings_csv, ids_file=args.list, output_root=args.output_root)

    mach = machine()
    df = to_records(timings_csv, mach)
    df.to_csv(os.path.join(out, "cdsm_records.csv"), index=False)
    with open(os.path.join(out, "cdsm_machine.json"), "w") as fh:
        json.dump(mach, fh, indent=2)

    print(f"\nMeasured on: {mach['cpu']} ({mach['model']}), "
          f"{mach['n_cores_perf']}P+{mach['n_cores_eff']}E cores, {mach['platform']}")
    for basis in ("full_pipeline", "headline_stage"):
        s = df[df.basis == basis]
        q1, me, q3 = s.wall_s.quantile([.25, .5, .75])
        print(f"  {basis:15s} n={len(s):3d}  median {me:6.3f}s  IQR {q1:.3f}-{q3:.3f}  "
              f"${s.cost_usd.median():.3e}/structure")
    print(f"\n-> {out}/cdsm_records.csv")


if __name__ == "__main__":
    main()
