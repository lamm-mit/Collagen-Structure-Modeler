#!/usr/bin/env python3
"""
ingest_af_server.py — turn downloaded AlphaFold Server results into scorer inputs.

AF Server emits 5 predictions per job (model_0 … model_4, ordered by ranking_score,
so model_0 is the top-ranked). We take model_0 — the standard way AF3 is reported,
and cleanly reproducible. This is a mild inconsistency with the single-unselected
sample used for Boltz/Chai/Protenix (AF3 gets confidence-best-of-5), but we measured
the effect and it is negligible: random vs confidence differs by ~0 on TM/lDDT and
~0.17 Å on backbone RMSD (well short of the oracle-best-of-5, confirming it is not
answer-cheating). Note as a methods footnote.

Point --results at the unzipped AF-Server download (a `folds_*` directory, or any
parent of the per-job folders). Re-runnable across days as new batches arrive.
Usage:
  python ingest_af_server.py                      # scans outputs/af3_msa/ for job folders
  python ingest_af_server.py --results <dir>
"""

import argparse
import glob
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs", "af3_msa")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=OUT_DIR,
                    help="dir containing AF-Server job folders (default: outputs/af3_msa/)")
    args = ap.parse_args()

    # top-ranked model of every job under the results dir (recursive)
    hits = glob.glob(os.path.join(args.results, "**", "fold_*_model_0.cif"), recursive=True)
    if not hits:
        raise SystemExit(f"no fold_*_model_0.cif found under {args.results}")

    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for src in sorted(hits):
        m = re.match(r"fold_(.+)_model_0\.cif$", os.path.basename(src))
        if not m:
            continue
        pdb = m.group(1).upper()
        shutil.copyfile(src, os.path.join(OUT_DIR, f"{pdb}_af3_msa.cif"))
        n += 1
        print(f"  {pdb}  ← model_0 (top-ranked)")
    print(f"\nIngested {n} AF3 structure(s) → {OUT_DIR}/<PDB>_af3_msa.cif")


if __name__ == "__main__":
    main()
