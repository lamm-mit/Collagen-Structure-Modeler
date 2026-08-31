#!/usr/bin/env python3
"""
make_af_server_json.py — build AlphaFold Server job JSON files for the 80 targets.

AlphaFold Server accepts up to 100 jobs per uploaded JSON but only runs 30/day, so
this writes stratified batches of 30/30/20 (each a representative homo/hetero mix)
plus a combined all-80 reference file. Upload one batch per day.

Encoding (validated against AF's server/example.json):
  - one job per PDB; homotrimer = one proteinChain with count:3, heterotrimer = three
    proteinChains count:1. Sequence is amino acids only (P at each HYP site).
  - hydroxyproline: modifications [{"ptmType":"CCD_HYP","ptmPosition":<1-indexed>}].
  - useStructureTemplate:false on every chain — our targets are deposited PDB entries,
    so templates (default on, up to 2025-02-03) would leak the answer and break the
    comparison with Boltz/Chai/Protenix (which used no templates).
  - modelSeeds:[] — AF Server assigns one random seed (its recommended default).

This mirrors the run_boltz/chai/protenix drivers' input builders; the actual AF3
prediction driver will live in this folder once the model weights arrive.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import targets as targets_mod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZES = [30, 30, 20]


def _chain(seq: str, count: int) -> dict:
    """An AF-Server proteinChain: 'P' at HYP sites + CCD_HYP mods (1-indexed),
    templates disabled to prevent structural leakage."""
    base = "".join("P" if c == "O" else c for c in seq)
    mods = [{"ptmType": "CCD_HYP", "ptmPosition": i + 1}
            for i, c in enumerate(seq) if c == "O"]
    chain = {"sequence": base, "count": count, "useStructureTemplate": False}
    if mods:
        chain["modifications"] = mods
    return {"proteinChain": chain}


def build_job(target) -> dict:
    if target.is_homotrimer:
        sequences = [_chain(target.sequences[0], 3)]
    else:
        sequences = [_chain(s, 1) for s in target.sequences]
    return {"name": target.pdb_id, "modelSeeds": [], "sequences": sequences,
            "dialect": "alphafoldserver", "version": 1}


def stratified_split(targets):
    """Split into batches of BATCH_SIZES, each proportionally homo/hetero."""
    homo = [t for t in targets if t.is_homotrimer]
    het = [t for t in targets if not t.is_homotrimer]
    n = len(targets)
    batches, hi, ei = [], 0, 0
    for k, size in enumerate(BATCH_SIZES):
        # proportional homo count for this batch (last batch takes the remainder)
        n_h = round(len(homo) * size / n) if k < len(BATCH_SIZES) - 1 else len(homo) - hi
        n_e = size - n_h
        batches.append(homo[hi:hi + n_h] + het[ei:ei + n_e])
        hi += n_h
        ei += n_e
    return batches


def main():
    targets = targets_mod.load_targets()
    jobs_all = [build_job(t) for t in targets]

    # sanity: HYP positions all land on 'O'
    bad = 0
    for t, job in zip(targets, jobs_all):
        seqs = t.sequences if not t.is_homotrimer else (t.sequences[0],)
        for orig, chain_entry in zip(seqs, job["sequences"]):
            for m in chain_entry["proteinChain"].get("modifications", []):
                if orig[m["ptmPosition"] - 1] != "O":
                    bad += 1
    assert bad == 0, f"{bad} HYP modifications not on an 'O'"

    # combined reference (all 80)
    with open(os.path.join(HERE, "af_server_all80.json"), "w") as fh:
        json.dump(jobs_all, fh, indent=2)

    batches = stratified_split(targets)
    print(f"{len(targets)} targets → batches of {BATCH_SIZES}  (HYP positions verified)")
    for k, batch in enumerate(batches, 1):
        jobs = [build_job(t) for t in batch]
        fn = f"af_server_batch{k}_{len(batch)}.json"
        with open(os.path.join(HERE, fn), "w") as fh:
            json.dump(jobs, fh, indent=2)
        n_h = sum(t.is_homotrimer for t in batch)
        n_hyp = sum(any(c == "O" for s in t.sequences for c in s) for t in batch)
        print(f"  {fn}: {len(batch)} jobs "
              f"({n_h} homo, {len(batch)-n_h} hetero, {n_hyp} with HYP) "
              f"— {[t.pdb_id for t in batch]}")


if __name__ == "__main__":
    main()
