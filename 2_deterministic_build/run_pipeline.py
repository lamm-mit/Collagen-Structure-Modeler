#!/usr/bin/env python3
"""
run_pipeline.py

Build collagen triple-helix structures from sequence and write four progressively
more complete outputs per PDB entry:

  gen_struct_coreonly/<PDB>_generated_core.cif
      backbone(core) -> side chains
      (pure THeBuScr: chains truncated to the common equal-length core)

  gen_struct_extendedchains/<PDB>_generated_extended.cif
      backbone(extend) -> side chains
      (unequal chains completed to their whole-triplet lengths)

  gen_struct_fullseq/<PDB>_generated_full.cif
      backbone(extend) -> terminal overhang -> side chains
      (deposited residue count restored at both termini)

  gen_struct_fullseq_reregistered/<PDB>_generated_full_reregistered.cif
      backbone(extend) -> terminal overhang -> register fix -> side chains
      (inter-chain stagger re-threaded to minimise terminal overhang)

All four are always produced. The only choice is which PDB IDs to build.

Usage (needs tleap + parmed, e.g. the MIT_environment conda env):
  python run_pipeline.py --pdb-id 8K4X
  python run_pipeline.py --list ids.txt
  python run_pipeline.py --all
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile

import step1_backbone_builder as step1
import step2_terminal_extension as step2
import step3_register_fix as step3
import step4_sidechain_builder as step4
import step5_relax as step5
import step5_anneal as step5a

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from data_locations import manifest_path  # noqa: E402

DEFAULT_MANIFEST = manifest_path("csv")
OUTPUT_ROOT = os.path.join(HERE, "outputs")   # builds are written locally

# THeBuScr's Phase-1 propensity math divides by zero on these (internal Gly-X-Y
# register interruptions); they fail before any output can be produced.
KNOWN_ZERODIV = {"1EI8", "6M80", "5K86", "7LXQ", "7LXP"}

# stage name -> (folder, file suffix)
OUTPUTS = {
    "core":               ("gen_struct_coreonly",             "generated_core"),
    "extended":           ("gen_struct_extendedchains",       "generated_extended"),
    "full":               ("gen_struct_fullseq",              "generated_full"),
    "full_reregistered":  ("gen_struct_fullseq_reregistered", "generated_full_reregistered"),
    "full_reregistered_relaxed": ("gen_struct_fullseq_reregistered_relaxed",
                                  "generated_full_reregistered_relaxed"),
    "full_reregistered_annealed": ("gen_struct_fullseq_reregistered_annealed",
                                   "generated_full_reregistered_annealed"),
}

# relax-method -> (relax callable, OUTPUTS key). Selectable so we can compare
# minimisation vs simulated annealing for the final side-chain relaxation. Both
# callables share the signature (prmtop, rst7, out_cif, traj_dcd=...).
RELAX_METHODS = {
    "minimize": (step5.relax, "full_reregistered_relaxed"),
    "anneal":   (step5a.anneal, "full_reregistered_annealed"),
}


def load_manifest(manifest_path):
    """Return {PDB_ID: [sequences]}. Single source of sequence truth for now;
    later this can be swapped to read the HuggingFace parquet — change only here."""
    out = {}
    with open(manifest_path, newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row["pdb_id"].strip().upper()
            if row["kind"].strip().lower() == "homotrimer":
                out[pid] = [row["chain_a_sequence"].strip().upper()]
            else:
                out[pid] = [row["chain_a_sequence"].strip().upper(),
                            row["chain_b_sequence"].strip().upper(),
                            row["chain_c_sequence"].strip().upper()]
    return out


def _sidechains(chains, stage, pdb_id):
    """Write `chains` backbone to a temp PDB, add side chains, save the stage CIF."""
    folder, suffix = OUTPUTS[stage]
    out_dir = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    out_cif = os.path.join(out_dir, f"{pdb_id}_{suffix}.cif")

    fd, backbone_pdb = tempfile.mkstemp(prefix=f"{pdb_id}_{stage}_", suffix="_bb.pdb")
    os.close(fd)
    try:
        step1.write_backbone_pdb(backbone_pdb, chains)
        step4.add_sidechains(backbone_pdb, out_cif)
    finally:
        if os.path.exists(backbone_pdb):
            os.remove(backbone_pdb)
    return out_cif


def _sidechains_and_relax(chains, pdb_id, relax_method):
    """Build the reregistered side-chain structure, then relax it (step5).

    `relax_method` selects minimisation (step5_relax) or simulated annealing
    (step5_anneal); the relaxed structure is routed to that method's output
    folder. Retains the tleap Amber files so step5 can use ff14SB (HYP already
    parameterised). Returns a short status note; a relaxation failure is
    non-fatal (the reregistered structure is still written)."""
    relax_fn, out_key = RELAX_METHODS[relax_method]
    rr_folder, rr_suffix = OUTPUTS["full_reregistered"]
    rx_folder, rx_suffix = OUTPUTS[out_key]
    rr_dir = os.path.join(OUTPUT_ROOT, rr_folder)
    rx_dir = os.path.join(OUTPUT_ROOT, rx_folder)
    os.makedirs(rr_dir, exist_ok=True)
    os.makedirs(rx_dir, exist_ok=True)
    rr_cif = os.path.join(rr_dir, f"{pdb_id}_{rr_suffix}.cif")
    rx_cif = os.path.join(rx_dir, f"{pdb_id}_{rx_suffix}.cif")
    traj = os.path.join(rx_dir, "trajectories", f"{pdb_id}.dcd")

    amber_work = tempfile.mkdtemp(prefix=f"{pdb_id}_amber_")
    amber_prefix = os.path.join(amber_work, "system")
    fd, backbone_pdb = tempfile.mkstemp(prefix=f"{pdb_id}_rereg_", suffix="_bb.pdb")
    os.close(fd)
    try:
        step1.write_backbone_pdb(backbone_pdb, chains)
        step4.add_sidechains(backbone_pdb, rr_cif, amber_out=amber_prefix)
        try:
            relax_fn(f"{amber_prefix}.prmtop", f"{amber_prefix}.rst7",
                     rx_cif, traj_dcd=traj)
            return f"{relax_method} OK"
        except Exception as e:  # noqa: BLE001 — relaxation is non-fatal
            return f"{relax_method} FAILED: {type(e).__name__}: {str(e)[:60]}"
    finally:
        if os.path.exists(backbone_pdb):
            os.remove(backbone_pdb)
        shutil.rmtree(amber_work, ignore_errors=True)


def build_all(pdb_id, sequences, relax_method="minimize"):
    """Produce all four outputs for one PDB ID. Returns (status, detail)."""
    try:
        core_chains = step1.build_backbone(sequences, mode="core")
        ext_chains = step1.build_backbone(sequences, mode="extend")
    except ZeroDivisionError:
        return "skipped", "THeBuScr div-by-zero (Gly-X-Y register interruption)"

    full_chains = step2.extend_termini(ext_chains, sequences)
    rereg_chains, shifts, improvement = step3.reregister(full_chains)

    _sidechains(core_chains, "core", pdb_id)
    _sidechains(ext_chains, "extended", pdb_id)
    _sidechains(full_chains, "full", pdb_id)

    # reregistered build, retaining its tleap Amber files so step5 can relax it
    relax_note = _sidechains_and_relax(rereg_chains, pdb_id, relax_method)

    reg_note = (f"re-threaded A/B/C={shifts} (−{improvement:.1f} A overhang)"
                if shifts != (0, 0, 0) else "register unchanged")
    return "ok", f"residues/chain={[len(c) for c in full_chains]}; {reg_note}; {relax_note}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb-id", help="single PDB ID to build")
    g.add_argument("--list", help="file with one PDB ID per line")
    g.add_argument("--all", action="store_true", help="build every ID in the manifest")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="manifest.csv path (default ../data/manifest.csv)")
    ap.add_argument("--relax-method", choices=tuple(RELAX_METHODS), default="minimize",
                    help="final side-chain relaxation: 'minimize' (step5_relax) or "
                         "'anneal' (step5_anneal simulated annealing). Routes to a "
                         "separate output folder so the two can be compared.")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.pdb_id:
        ids = [args.pdb_id.strip().upper()]
    elif args.list:
        with open(args.list) as fh:
            ids = [ln.strip().upper() for ln in fh if ln.strip()]
    else:
        ids = list(manifest)

    missing = [i for i in ids if i not in manifest]
    if missing:
        sys.exit(f"ERROR: PDB ID(s) not in manifest: {missing}")

    print(f"Building {len(ids)} structure(s), 4 outputs each "
          f"(relax method: {args.relax_method}).\n")
    n_ok = 0
    for pid in ids:
        status, detail = build_all(pid, manifest[pid], relax_method=args.relax_method)
        mark = {"ok": "✓", "skipped": "–"}.get(status, "✗")
        print(f"  {mark} {pid} {status}: {detail}")
        n_ok += status == "ok"
    print(f"\nDone. {n_ok}/{len(ids)} built (×4 outputs).")


if __name__ == "__main__":
    main()
