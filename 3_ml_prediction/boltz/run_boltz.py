#!/usr/bin/env python3
"""
run_boltz.py — drive Boltz-2 structure predictions for the benchmark targets.

For each target (from ../shared/targets.py) this builds a Boltz input JSON,
submits it via the `boltz-api` CLI, waits for the result, and copies the
predicted CIF into outputs/boltz/<PDB>_boltz.cif for scoring.

Boltz input encoding (validated against the API):
  - protein entity per distinct chain sequence; homotrimer = one entity with
    chain_ids ["A","B","C"], heterotrimer = three entities (A/B/C).
  - hydroxyproline: base sequence carries 'P' at each HYP site, plus a
    modification {"type":"ccd","residue_index":<0-indexed>,"value":"HYP"}.
  - MSA mode is selectable (--msa empty|auto); empty = single-sequence.

Resume / cost safety: each submission uses --idempotency-key <PDB_ID>, so a
re-run does not resubmit or re-bill a target that already ran. A local state.csv
also records status. Estimate the batch cost first with --estimate.

Requires: `boltz-api` on PATH and an authenticated session (`boltz-api auth login`).
Usage:
  python run_boltz.py --estimate --all           # cost only, no GPU/credits
  python run_boltz.py --pdb-id 8YUK --msa empty
  python run_boltz.py --all --msa empty
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import targets as targets_mod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
OUT_DIR = os.path.join(HERE, "..", "outputs", "boltz")
STATE_CSV = os.path.join(HERE, "state.csv")
MODEL = "boltz-2.1"
BOLTZ = shutil.which("boltz-api") or os.path.expanduser("~/.local/bin/boltz-api")

# relative path of the predicted CIF inside a run directory
PRED_CIF = os.path.join("outputs", "files", "prediction", "sample_0_predicted_structure.cif")


# ── input encoding ───────────────────────────────────────────────────────────
def _entity(seq: str, chain_ids: list, msa_mode: str) -> dict:
    base = "".join("P" if c == "O" else c for c in seq)          # HYP parent = Pro
    mods = [{"type": "ccd", "residue_index": i, "value": "HYP"}   # residue_index is 0-indexed
            for i, c in enumerate(seq) if c == "O"]
    ent = {"type": "protein", "value": base, "chain_ids": chain_ids, "modifications": mods}
    # Valid msa.type values are only "empty" and "custom"; automatic MSA generation
    # is the default and is triggered by OMITTING the msa field.
    if msa_mode == "empty":
        ent["msa"] = {"type": "empty"}
    return ent


def build_input(target, msa_mode: str) -> dict:
    """Boltz JSON payload for a target (homo- or heterotrimer)."""
    if target.is_homotrimer:
        entities = [_entity(target.sequences[0], ["A", "B", "C"], msa_mode)]
    else:
        entities = [_entity(s, [c], msa_mode)
                    for s, c in zip(target.sequences, "ABC")]
    return {"entities": entities}


# ── CLI wrapper ──────────────────────────────────────────────────────────────
def _boltz(*args) -> subprocess.CompletedProcess:
    return subprocess.run([BOLTZ, *args], capture_output=True, text=True)


def estimate_cost(payload: dict) -> float:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh); path = fh.name
    try:
        r = _boltz("predictions:structure-and-binding", "estimate-cost",
                   "--model", MODEL, "--input", f"@json://{path}")
        try:
            return float(json.loads(r.stdout)["estimated_cost_usd"])
        except Exception:
            return float("nan")
    finally:
        os.remove(path)


def predict(target, msa_mode: str, tag: str = "") -> str:
    """Submit + wait + download one target; return the output CIF path or raise.

    `tag` routes output to a distinct folder and idempotency key (used to compare
    MSA modes side by side, so the two runs don't share a cached result)."""
    payload = build_input(target, msa_mode)
    name = f"{target.pdb_id}_{tag}" if tag else target.pdb_id
    idem = f"{target.pdb_id}-{tag}" if tag else target.pdb_id
    out_dir = os.path.join(HERE, "..", "outputs", f"boltz_{tag}" if tag else "boltz")
    run_dir = os.path.join(RAW_DIR, name)
    shutil.rmtree(run_dir, ignore_errors=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh); path = fh.name
    try:
        r = _boltz("predictions:structure-and-binding", "run",
                   "--model", MODEL, "--input", f"@json://{path}",
                   "--idempotency-key", idem,              # resume / no double-billing
                   "--name", name, "--root-dir", RAW_DIR)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:200] or r.stdout.strip()[:200])
    finally:
        os.remove(path)

    src = os.path.join(run_dir, PRED_CIF)
    if not os.path.exists(src):
        raise RuntimeError(f"no predicted CIF at {src}")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{target.pdb_id}_boltz.cif")
    shutil.copyfile(src, dst)
    return dst


# ── state ────────────────────────────────────────────────────────────────────
def _write_state(rows):
    with open(STATE_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "status", "detail"])
        w.writeheader(); w.writerows(rows)


def _select(args, all_targets):
    if args.pdb_id:
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
    g.add_argument("--pdb-id"); g.add_argument("--list"); g.add_argument("--all", action="store_true")
    ap.add_argument("--msa", choices=("empty", "auto"), default="empty",
                    help="MSA mode (default empty = single-sequence)")
    ap.add_argument("--tag", default="",
                    help="route output to outputs/boltz_<tag>/ with a distinct "
                         "idempotency key (for comparing MSA modes side by side)")
    ap.add_argument("--estimate", action="store_true",
                    help="only estimate total cost (no GPU/credits spent)")
    args = ap.parse_args()

    ts = _select(args, targets_mod.load_targets())
    if not ts:
        sys.exit("no matching targets")

    if args.estimate:
        total = 0.0
        for t in ts:
            c = estimate_cost(build_input(t, args.msa))
            total += c if c == c else 0
        print(f"Estimated cost for {len(ts)} target(s), msa={args.msa}: ${total:.2f} "
              f"(${total/len(ts):.4f}/structure)")
        return

    print(f"Predicting {len(ts)} target(s) with Boltz-2 (msa={args.msa}) …")
    rows, ok = [], 0
    for t in ts:
        try:
            dst = predict(t, args.msa, tag=args.tag)
            rows.append({"pdb_id": t.pdb_id, "status": "ok", "detail": os.path.basename(dst)})
            ok += 1
            print(f"  ✓ {t.pdb_id}")
        except Exception as e:  # noqa: BLE001
            rows.append({"pdb_id": t.pdb_id, "status": "error", "detail": str(e)[:120]})
            print(f"  ✗ {t.pdb_id}: {str(e)[:120]}")
    _write_state(rows)
    print(f"\nDone. {ok}/{len(ts)} predicted → {OUT_DIR}")


if __name__ == "__main__":
    main()
