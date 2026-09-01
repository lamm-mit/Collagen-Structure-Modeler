#!/usr/bin/env python3
"""
score_seed_sensitivity.py — accuracy of the seed-sweep predictions.

The accuracy benchmark used one diffusion sample from one seed per model
(Chai 42, Protenix 1, Boltz 42, AF3 1). The compute-cost seed sweep runs
5 seeds x 10 length-stratified targets per model for timing, and keeps every
CIF (bench_common.save_cif: "keeping them costs nothing and means the
seed-variance question can be answered on these targets later"). This scores
those structures against the experimental references, identically to
4_scoring/score.py, and answers the follow-up: do the CDSM-vs-model accuracy
conclusions depend on the seed? If not, re-running all 80 targets x 5 seeds
buys nothing.

Tiers: bare dirs (af3/, boltz/, ...) are the L40S tier of record; *_L4 dirs
are the cheaper-tier replication sweep. Hardware should not change accuracy
beyond GPU nondeterminism, so L4 doubles as a second independent replicate.

Outputs (4_scoring/compute_cost/results/):
    seed_sensitivity_scores.csv    one row per (model, tier, seed, pdb_id)

Usage:
    python 3_ml_prediction/shared/score_seed_sensitivity.py [--tier L40S]
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "4_scoring"))
sys.path.insert(0, REPO)

import score as S  # noqa: E402
from data_locations import local_out, scores_dir  # noqa: E402

RESULTS = local_out("4_scoring", "compute_cost", "results")
SEED_CIFS = os.path.join(RESULTS, "seed_cifs")
OUT_CSV = os.path.join(RESULTS, "seed_sensitivity_scores.csv")

METRICS = ["tm_score", "global_lddt_allatom",
           "global_rmsd_backbone", "global_rmsd_allatom"]
CDSM_VARIANT = "fullseq_reregistered_relaxed"   # the headline CDSM stage


def score_sweeps(tier=None):
    rows = []
    for d in sorted(os.listdir(SEED_CIFS)):
        model, t = (d[:-3], "L4") if d.endswith("_L4") else (d, "L40S")
        if tier and t != tier:
            continue
        for sd in sorted(os.listdir(os.path.join(SEED_CIFS, d))):
            seed = int(sd.split("_")[1])
            for f in sorted(os.listdir(os.path.join(SEED_CIFS, d, sd))):
                if not f.endswith(".cif"):
                    continue
                pid = f.split("_")[0]
                ref = os.path.join(S.EXP_DIR, f"{pid}.cif")
                pred = os.path.join(SEED_CIFS, d, sd, f)
                try:
                    summary, _ = S.score_one(pred, ref)
                except Exception as e:  # noqa: BLE001
                    print(f"    x {d} seed {seed} {pid}: {type(e).__name__}: {e}")
                    continue
                rows.append({"model": model, "tier": t, "seed": seed,
                             "pdb_id": pid, **summary})
                print(f"    {d:12s} seed {seed} {pid}: "
                      f"TM={summary['tm_score']:.4f} "
                      f"lDDT_aa={summary['global_lddt_allatom']:.4f} "
                      f"RMSD_bb={summary['global_rmsd_backbone']:.2f} A")
    return rows


def published_scores():
    """scores_summary.csv, from the Hub if the working tree shadows it with a
    partial results/ directory (data_locations prefers the working tree)."""
    path = os.path.join(scores_dir(), "scores_summary.csv")
    if not os.path.exists(path):
        os.environ["COLLAGEN_FORCE_HUB"] = "1"
        path = os.path.join(scores_dir(), "scores_summary.csv")
    return pd.read_csv(path)


def analyse(df):
    print("\n== per-seed medians across the 10 targets (tier of record) ==")
    t0 = df[df.tier == "L40S"]
    med = t0.groupby(["model", "seed"])[METRICS].median().round(4)
    print(med.to_string())

    print("\n== spread across seeds, per target: RMSD backbone (A) ==")
    piv = t0.pivot_table(index=["model", "pdb_id"], columns="seed",
                         values="global_rmsd_backbone")
    piv["sd"] = piv.std(axis=1)
    piv["range"] = piv.max(axis=1) - piv.min(axis=1)
    print(piv.round(3).to_string())
    print("\nacross-target summary of that spread, per model:")
    print(piv.groupby("model")[["sd", "range"]].median().round(4).to_string())

    print("\n== CDSM (deterministic, no seed) vs each model, per seed ==")
    pub = published_scores()
    cdsm = (pub[pub.variant == CDSM_VARIANT]
            .set_index("pdb_id")[METRICS])
    ids = sorted(set(t0.pdb_id) & set(cdsm.index))
    cdsm = cdsm.loc[ids]
    print(f"CDSM {CDSM_VARIANT} on the same {len(ids)} targets, median: "
          + ", ".join(f"{m}={cdsm[m].median():.4f}" for m in METRICS))
    for (model, seed), g in t0.groupby(["model", "seed"]):
        g = g.set_index("pdb_id").loc[ids]
        # lower RMSD-bb is better: count targets where the model beats CDSM
        wins = int((g["global_rmsd_backbone"] < cdsm["global_rmsd_backbone"]).sum())
        print(f"  {model:9s} seed {seed}: median RMSD_bb "
              f"{g['global_rmsd_backbone'].median():6.3f} A vs CDSM "
              f"{cdsm['global_rmsd_backbone'].median():6.3f} A; "
              f"model beats CDSM on {wins}/{len(ids)} targets")

    if "L4" in set(df.tier):
        print("\n== L4 replication: per-target |L4 - L40S| RMSD-bb, matched "
              "(model, seed, target) ==")
        l4 = df[df.tier == "L4"].set_index(["model", "seed", "pdb_id"])
        l40 = t0.set_index(["model", "seed", "pdb_id"])
        both = l40.join(l4, lsuffix="_l40s", rsuffix="_l4", how="inner")
        d = (both["global_rmsd_backbone_l4"]
             - both["global_rmsd_backbone_l40s"]).abs()
        print(f"  n={len(d)}  median {d.median():.4f} A  max {d.max():.4f} A")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["L40S", "L4"], default=None,
                    help="score only one tier (default: both)")
    ap.add_argument("--analyse-only", action="store_true",
                    help="skip scoring; re-run the analysis on the existing CSV")
    args = ap.parse_args()

    if not args.analyse_only:
        rows = score_sweeps(tier=args.tier)
        df = pd.DataFrame(rows)
        if os.path.exists(OUT_CSV) and args.tier:
            old = pd.read_csv(OUT_CSV)
            df = pd.concat([old[old.tier != args.tier], df], ignore_index=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n-> {OUT_CSV} ({len(df)} rows)")
    else:
        df = pd.read_csv(OUT_CSV)

    analyse(df)


if __name__ == "__main__":
    main()
