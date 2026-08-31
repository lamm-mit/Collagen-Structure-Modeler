#!/usr/bin/env python3
"""
run_chai.py — drive Chai-1 structure predictions for the benchmark targets.

For each target (from ../shared/targets.py) this builds a Chai-1 FASTA, runs it
on a Modal cloud GPU via chai_modal_app.fold, and writes the top-ranked model to
outputs/chai/<PDB>_chai.cif for scoring — the same contract as run_boltz.py.

Chai-1 input encoding (validated against the Chai FASTA spec):
  - one FASTA record per physical chain: `>protein|name=A` … A/B/C.
    homotrimer  = 3 records with the SAME sequence (chain_sequences() expands it);
    heterotrimer = 3 records with the three distinct sequences.
  - hydroxyproline: each `O` in the sequence is written inline as the CCD code
    `(HYP)` (Chai's modified-residue syntax, e.g. `...P(HYP)G...`).
  - MSA: default single-sequence (ESM embeddings, no MSA). --msa auto turns on
    Chai's MSA server. Single-sequence is preferred for collagen (cf. Boltz).

Resume: a target whose outputs/chai/<PDB>_chai.cif already exists is skipped, so
re-runs only fold what's missing. A local state.csv records status.

Prerequisite (one-time, needs a browser — run it yourself):
    python3.12 -m modal setup

Usage (base python3.12, which has modal + gemmi):
  python run_chai.py --smoke 8YUK        # 1 HYP structure + validate HYP positions
  python run_chai.py --pdb-id 1BKV
  python run_chai.py --all               # all 80, resumes, parallel across GPUs
  python run_chai.py --all --force       # re-fold even if output exists
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import targets as targets_mod          # noqa: E402
import modal                           # noqa: E402
from chai_modal_app import app, fold   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs", "chai")
STATE_CSV = os.path.join(HERE, "state.csv")

THREE2ONE_HYP = {"HYP": "O"}           # for smoke validation of round-tripped HYP


# ── input encoding ───────────────────────────────────────────────────────────
def build_fasta(target) -> str:
    """Chai-1 FASTA for a target: 3 protein records A/B/C, HYP written as (HYP)."""
    records = []
    for chain_id, seq in zip("ABC", target.chain_sequences()):
        chai_seq = "".join("(HYP)" if c == "O" else c for c in seq)
        records.append(f">protein|name={chain_id}\n{chai_seq}")
    return "\n".join(records) + "\n"


# ── output ───────────────────────────────────────────────────────────────────
def _out_path(pdb_id: str) -> str:
    return os.path.join(OUT_DIR, f"{pdb_id}_chai.cif")


def _write(pdb_id: str, cif_text: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = _out_path(pdb_id)
    with open(dst, "w") as fh:
        fh.write(cif_text)
    return dst


# ── HYP smoke validation ─────────────────────────────────────────────────────
def validate_hyp(target, cif_path: str) -> str:
    """Confirm the predicted CIF has 3 chains and HYP at the right positions.

    Rebuilds each chain's one-letter sequence from the CIF (HYP→O) and checks it
    matches the target sequence. Returns "" if OK, else a human-readable reason.
    This is the step that caught Boltz's 0/1-indexing bug — repeat it per model.
    """
    import gemmi

    one = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
           "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
           "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
           "TYR": "Y", "VAL": "V", "HYP": "O"}
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    chains = []
    for chain in st[0]:
        seq = "".join(one.get(r.name, "") for r in chain if r.name in one and r.find_atom("CA", "*"))
        if seq:
            chains.append(seq)
    if len(chains) != 3:
        return f"expected 3 chains, got {len(chains)}"

    expected = [s for s in target.chain_sequences()]
    # match each predicted chain to an expected sequence (order/id may differ)
    exp_pool = list(expected)
    for pred in chains:
        if pred in exp_pool:
            exp_pool.remove(pred)
        else:
            return (f"chain sequence mismatch (HYP misplaced?)\n"
                    f"       predicted: {pred}\n"
                    f"       expected any of: {expected}")
    n_hyp_expected = sum(s.count("O") for s in expected)
    n_hyp_pred = sum(s.count("O") for s in chains)
    if n_hyp_expected != n_hyp_pred:
        return f"HYP count mismatch: expected {n_hyp_expected}, got {n_hyp_pred}"
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
                    help="empty = single-sequence (default); auto = Chai MSA server")
    ap.add_argument("--force", action="store_true", help="re-fold even if output exists")
    args = ap.parse_args()

    use_msa = args.msa == "auto"
    ts = _select(args, targets_mod.load_targets())
    if not ts:
        sys.exit("no matching targets")

    # resume: drop targets already predicted (unless --force or --smoke)
    if not args.force and not args.smoke:
        pending = [t for t in ts if not os.path.exists(_out_path(t.pdb_id))]
        skipped = len(ts) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-predicted target(s); {len(pending)} to fold.")
        ts = pending
    if not ts:
        print("Nothing to do — all selected targets already have outputs.")
        return

    print(f"Folding {len(ts)} target(s) with Chai-1 on Modal (msa={args.msa}) …")
    rows, ok = [], 0
    with modal.enable_output(), app.run():
        if args.smoke:
            t = ts[0]
            cif = fold.remote(t.pdb_id, build_fasta(t), use_msa)
            dst = _write(t.pdb_id, cif)
            reason = validate_hyp(t, dst)
            if reason:
                print(f"  ✗ {t.pdb_id}: HYP validation FAILED — {reason}")
                sys.exit(1)
            print(f"  ✓ {t.pdb_id}: 3 chains, HYP positions correct → {dst}")
            return

        names = [t.pdb_id for t in ts]
        fastas = [build_fasta(t) for t in ts]
        results = fold.map(names, fastas, [use_msa] * len(ts),
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
