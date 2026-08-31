#!/usr/bin/env python3
"""
run_protenix.py — drive Protenix structure predictions for the benchmark targets.

Same contract as run_boltz.py / run_chai.py: for each target (from
../shared/targets.py) build a Protenix JSON job, run it on a Modal cloud GPU via
protenix_modal_app.fold, and write the top-ranked model to
outputs/protenix/<PDB>_protenix.cif for scoring.

Protenix input encoding (validated against docs/infer_json_format.md):
  - one proteinChain per DISTINCT sequence; homotrimer = a single chain with
    "count": 3, heterotrimer = three chains each "count": 1.
  - hydroxyproline: the "sequence" carries 'P' at each HYP site (parent proline),
    plus a modification {"ptmType": "CCD_HYP", "ptmPosition": <1-INDEXED>}.
    NB the CCD_ prefix and 1-indexing (Boltz used bare "HYP" and 0-indexing).
  - MSA: default single-sequence (--use_msa false), preferred for collagen.

Resume: a target whose output CIF already exists is skipped. state.csv records status.

Prerequisite (one-time, browser): python3.12 -m modal setup
Usage (base python3.12, which has modal + gemmi):
  python run_protenix.py --smoke 8YUK      # 1 HYP structure + validate HYP positions
  python run_protenix.py --pdb-id 1BKV
  python run_protenix.py --all             # all 80, resumes, parallel across GPUs
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import targets as targets_mod              # noqa: E402
import modal                               # noqa: E402
from protenix_modal_app import app, fold   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs", "protenix")
STATE_CSV = os.path.join(HERE, "state.csv")


# ── input encoding ───────────────────────────────────────────────────────────
def _chain(seq: str, count: int) -> dict:
    """A Protenix proteinChain: 'P' at HYP sites + CCD_HYP mods (1-indexed)."""
    base = "".join("P" if c == "O" else c for c in seq)
    mods = [{"ptmType": "CCD_HYP", "ptmPosition": i + 1}      # 1-indexed
            for i, c in enumerate(seq) if c == "O"]
    chain = {"sequence": base, "count": count}
    if mods:
        chain["modifications"] = mods
    return {"proteinChain": chain}


def build_job(target) -> dict:
    """Protenix input record for a target (homo- or heterotrimer)."""
    if target.is_homotrimer:
        sequences = [_chain(target.sequences[0], 3)]         # one chain, 3 copies
    else:
        sequences = [_chain(s, 1) for s in target.sequences]  # three distinct chains
    return {"name": target.pdb_id, "sequences": sequences}


# ── output ───────────────────────────────────────────────────────────────────
def _out_path(pdb_id: str) -> str:
    return os.path.join(OUT_DIR, f"{pdb_id}_protenix.cif")


def _write(pdb_id: str, cif_text: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = _out_path(pdb_id)
    with open(dst, "w") as fh:
        fh.write(cif_text)
    return dst


# ── HYP smoke validation ─────────────────────────────────────────────────────
def validate_hyp(target, cif_path: str) -> str:
    """Confirm the predicted CIF has 3 chains and HYP at the right positions.
    Returns "" if OK, else a human-readable reason (mirrors run_chai.validate_hyp)."""
    import gemmi

    one = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
           "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
           "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
           "TYR": "Y", "VAL": "V", "HYP": "O"}
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    chains = []
    for chain in st[0]:
        seq = "".join(one.get(r.name, "") for r in chain
                      if r.name in one and r.find_atom("CA", "*"))
        if seq:
            chains.append(seq)
    if len(chains) != 3:
        return f"expected 3 chains, got {len(chains)}"

    expected = list(target.chain_sequences())
    pool = list(expected)
    for pred in chains:
        if pred in pool:
            pool.remove(pred)
        else:
            return (f"chain sequence mismatch (HYP misplaced?)\n"
                    f"       predicted: {pred}\n"
                    f"       expected any of: {expected}")
    n_exp = sum(s.count("O") for s in expected)
    n_pred = sum(s.count("O") for s in chains)
    if n_exp != n_pred:
        return f"HYP count mismatch: expected {n_exp}, got {n_pred}"
    return ""


# ── state ────────────────────────────────────────────────────────────────────
def _write_state(rows):
    with open(STATE_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "status", "detail"])
        w.writeheader()
        w.writerows(rows)


def _select(args, all_targets):
    if args.smoke:
        want = {args.smoke.strip().upper()}
    elif args.pdb_id:
        want = {args.pdb_id.strip().upper()}
    elif args.list:
        want = {l.strip().upper() for l in open(args.list) if l.strip()}
    else:
        want = None
    return [t for t in all_targets if want is None or t.pdb_id in want]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", metavar="PDB", help="fold 1 target and validate HYP positions")
    g.add_argument("--pdb-id")
    g.add_argument("--list")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--msa", choices=("empty", "auto"), default="empty",
                    help="empty = single-sequence (default); auto = Protenix MSA search")
    ap.add_argument("--force", action="store_true", help="re-fold even if output exists")
    args = ap.parse_args()

    use_msa = args.msa == "auto"
    ts = _select(args, targets_mod.load_targets())
    if not ts:
        sys.exit("no matching targets")

    if not args.force and not args.smoke:
        pending = [t for t in ts if not os.path.exists(_out_path(t.pdb_id))]
        skipped = len(ts) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-predicted target(s); {len(pending)} to fold.")
        ts = pending
    if not ts:
        print("Nothing to do — all selected targets already have outputs.")
        return

    print(f"Folding {len(ts)} target(s) with Protenix on Modal (msa={args.msa}) …")
    rows, ok = [], 0
    with modal.enable_output(), app.run():
        if args.smoke:
            t = ts[0]
            cif = fold.remote(t.pdb_id, build_job(t), use_msa)
            dst = _write(t.pdb_id, cif)
            reason = validate_hyp(t, dst)
            if reason:
                print(f"  ✗ {t.pdb_id}: HYP validation FAILED — {reason}")
                sys.exit(1)
            print(f"  ✓ {t.pdb_id}: 3 chains, HYP positions correct → {dst}")
            return

        names = [t.pdb_id for t in ts]
        jobs = [build_job(t) for t in ts]
        results = fold.map(names, jobs, [use_msa] * len(ts),
                           return_exceptions=True, order_outputs=True)
        for t, res in zip(ts, results):
            if isinstance(res, Exception):
                rows.append({"pdb_id": t.pdb_id, "status": "error", "detail": str(res)[:120]})
                print(f"  ✗ {t.pdb_id}: {str(res)[:120]}")
                continue
            dst = _write(t.pdb_id, res)
            rows.append({"pdb_id": t.pdb_id, "status": "ok", "detail": os.path.basename(dst)})
            ok += 1
            print(f"  ✓ {t.pdb_id}")

    _write_state(rows)
    print(f"\nDone. {ok}/{len(ts)} predicted → {OUT_DIR}")


if __name__ == "__main__":
    main()
