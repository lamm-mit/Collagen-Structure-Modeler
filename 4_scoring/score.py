#!/usr/bin/env python3
"""
score.py — score generated collagen structures against the experimental ones.

Scores the two final variants:
  new_structure_pipeline/gen_struct_fullseq/<PDB>_generated_full.cif
  new_structure_pipeline/gen_struct_fullseq_reregistered/<PDB>_generated_full_reregistered.cif
against
  data/experimental_cif/<PDB>.cif

Metrics per structure (each of RMSD and lDDT computed both all-atom and
backbone-only; TM-score is Cα by construction):
  - Global RMSD            (all-atom, backbone)   in USalign's global-fit frame
  - Per-residue RMSD       (all-atom, backbone)   same frame, written per residue
  - Global lDDT            (all-atom, backbone)   superposition-free, pooled pairs
  - Per-residue lDDT       (all-atom, backbone)   superposition-free, per residue
  - TM-score               (Cα, reference-normalised, US-align -mm 1)
  - coverage               (matched ref residues / ref residues)

Outputs (into scoring/):
  scores_summary.csv       one row per (PDB, variant): all global scores + coverage
  scores_per_residue.csv   long format: PDB, variant, chain, resnum, per-residue values

Needs: gemmi, numpy, and US-align at ../tools/USalign. No tleap/parmed.
Run:   python score.py            # all structures in both variant folders
       python score.py --pdb-id 8K4X
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys

import gemmi
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_locations import (  # noqa: E402
    cdsm_dir, experimental_cif_dir, local_out, predictions_dir, repo_root)

RESULTS = local_out("4_scoring", "results")          # written here, uploaded later
USALIGN = os.path.join(repo_root(), "tools", "USalign")
EXP_DIR = experimental_cif_dir()

# Directories are resolved lazily on first use: scoring only af3_msa shouldn't pull
# the CDSM stages down from HuggingFace.
VARIANTS = {
    "coreonly":             (lambda: cdsm_dir("coreonly"),
                             "_generated_core"),
    "fullseq":              (lambda: cdsm_dir("fullseq"),
                             "_generated_full"),
    "fullseq_reregistered": (lambda: cdsm_dir("fullseq_reregistered"),
                             "_generated_full_reregistered"),
    "fullseq_reregistered_relaxed": (
        lambda: cdsm_dir("fullseq_reregistered_relaxed"),
        "_generated_full_reregistered_relaxed"),
    # ML predictions
    "boltz":    (lambda: predictions_dir("boltz"), "_boltz"),
    "chai":     (lambda: predictions_dir("chai"), "_chai"),
    "protenix": (lambda: predictions_dir("protenix"), "_protenix"),
    "af3_msa":   (lambda: predictions_dir("af3_msa"), "_af3_msa"),
    "af3_nomsa": (lambda: predictions_dir("af3_nomsa"), "_af3_nomsa"),
}

BACKBONE = ("N", "CA", "C", "O")
LDDT_R0 = 15.0
LDDT_THRESH = (0.5, 1.0, 2.0, 4.0)

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "HYP": "O",
    # Amber/ff14SB protonation- and bond-state variants (tleap renames these in
    # the generated structures) — mapped to their standard residue letter so the
    # sequence matches the experimental HIS/CYS/ASP/GLU/LYS.
    "HID": "H", "HIE": "H", "HIP": "H", "HISD": "H", "HISE": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}


# ── parsing ─────────────────────────────────────────────────────────────────
def parse_cif(path):
    """{chain: [ {resname, resnum, seq1, atoms:{name:(x,y,z)}} ]} for protein
    residues that carry a CA. Heavy atoms only (H skipped defensively)."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = {}
    for chain in st[0]:
        residues = []
        for res in chain:
            if res.name not in THREE2ONE:
                continue
            atoms = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z])
                     for a in res if a.element != gemmi.Element("H")}
            if "CA" not in atoms:
                continue
            residues.append({"resname": res.name, "resnum": res.seqid.num,
                             "seq1": THREE2ONE[res.name], "atoms": atoms})
        if residues:
            out[chain.name] = residues
    return out


def chain_seq(residues):
    return "".join(r["seq1"] for r in residues)


def seq_match(s_model, s_ref):
    """(model_idx, ref_idx) mapping; a prediction is (a subsequence of / equal to)
    the reference. None if no ≥3-residue common prefix."""
    if s_model == s_ref:
        return list(range(len(s_model))), list(range(len(s_ref)))
    p = s_ref.find(s_model)
    if p >= 0:
        return list(range(len(s_model))), list(range(p, p + len(s_model)))
    p = s_model.find(s_ref)
    if p >= 0:
        return list(range(p, p + len(s_ref))), list(range(len(s_ref)))
    n = min(len(s_model), len(s_ref))
    pre = 0
    while pre < n and s_model[pre] == s_ref[pre]:
        pre += 1
    return (list(range(pre)), list(range(pre))) if pre >= 3 else None


# ── US-align (TM-score + superposition matrix) ──────────────────────────────
def run_usalign(pred_path, ref_path):
    """Return (tm_score_ref_normalised, (t, u)) where X = t + u·x maps the
    prediction onto the reference. None on failure."""
    mfile = pred_path + ".usmat"
    try:
        out = subprocess.run(
            [USALIGN, pred_path, ref_path, "-mm", "1", "-ter", "1", "-m", mfile],
            capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return None, None

    tm = None
    for line in out.splitlines():
        if "normalized by length of Structure_2" in line:
            m = re.search(r"TM-score= *([0-9.]+)", line)
            tm = float(m.group(1)) if m else None
    t, u = None, None
    if os.path.exists(mfile):
        rows = []
        for line in open(mfile):
            m = re.match(r"\s*([012])\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)", line)
            if m:
                rows.append([float(x) for x in m.groups()[1:]])
        os.remove(mfile)
        if len(rows) == 3:
            arr = np.array(rows)
            t = arr[:, 0]
            u = arr[:, 1:]
    return tm, (t, u)


def superpose(struct, t, u):
    return {c: [{**r, "atoms": {n: t + u @ xyz for n, xyz in r["atoms"].items()}}
                for r in res] for c, res in struct.items()}


# ── chain mapping ───────────────────────────────────────────────────────────
def best_chain_map(pred, ref, superposed=True):
    """Assign pred chains to ref chains.

    Primary criterion: maximise the total number of matched residues (sequence),
    so near-identical chains of *different length* are paired correctly rather
    than by a geometric fluke. Ties (e.g. a true homotrimer, where every mapping
    matches fully) are broken by minimum mean CA distance when `superposed`, which
    selects the correct coaxial assignment. Returns [(pred_chain, ref_chain), ...].
    """
    pc, rc = list(pred), list(ref)
    if len(pc) != len(rc):
        return list(zip(pc, rc))  # size-mismatch fallback: pair by order
    best = None  # (−matched, cost)
    for perm in itertools.permutations(rc):
        matched = 0
        cost = 0.0
        for p, r in zip(pc, perm):
            m = seq_match(chain_seq(pred[p]), chain_seq(ref[r]))
            if m is None:
                continue
            mi, ri = m
            matched += len(ri)
            if superposed:
                d = [np.linalg.norm(pred[p][a]["atoms"]["CA"] - ref[r][b]["atoms"]["CA"])
                     for a, b in zip(mi, ri)]
                cost += float(np.sum(d))
        key = (-matched, cost)  # most residues matched, then tightest packing
        if best is None or key < best[0]:
            best = (key, list(zip(pc, perm)))
    return best[1] if best else list(zip(pc, rc))


# ── RMSD (in the US-align frame) ────────────────────────────────────────────
def rmsd_scores(pred_sup, ref, chain_map, atom_names=None):
    """Global + per-residue RMSD over matched atoms. atom_names=None → all shared
    heavy atoms; else restrict to that set (e.g. backbone). Per-residue keyed by
    (ref_chain, ref_resnum)."""
    per_res = {}
    sq_all, n_all = 0.0, 0
    for p, r in chain_map:
        m = seq_match(chain_seq(pred_sup[p]), chain_seq(ref[r]))
        if m is None:
            continue
        for a, b in zip(*m):
            pa, rb = pred_sup[p][a]["atoms"], ref[r][b]["atoms"]
            names = set(pa) & set(rb)
            if atom_names is not None:
                names &= set(atom_names)
            if not names:
                continue
            sq = sum(float(np.sum((pa[n] - rb[n]) ** 2)) for n in names)
            per_res[(r, ref[r][b]["resnum"])] = (sq / len(names)) ** 0.5
            sq_all += sq
            n_all += len(names)
    g = (sq_all / n_all) ** 0.5 if n_all else float("nan")
    return g, per_res


# ── lDDT (superposition-free) ───────────────────────────────────────────────
def _atoms_flat(struct, chain_map, ref_key, atom_names):
    """Flatten matched atoms into (coords, res_id list) for model and ref, in a
    consistent order. res_id = (ref_chain, ref_resnum)."""
    mp, rp, rid = [], [], []
    for p, r in chain_map:
        m = seq_match(chain_seq(struct["pred"][p]), chain_seq(struct["ref"][r]))
        if m is None:
            continue
        for a, b in zip(*m):
            pa = struct["pred"][p][a]["atoms"]
            rb = struct["ref"][r][b]["atoms"]
            names = set(pa) & set(rb)
            if atom_names is not None:
                names &= set(atom_names)
            for n in sorted(names):
                mp.append(pa[n]); rp.append(rb[n])
                rid.append((r, struct["ref"][r][b]["resnum"]))
    return np.array(mp), np.array(rp), rid


def lddt_scores(pred, ref, atom_names=None):
    """Global (pooled) + per-residue lDDT, maximised over chain permutations.
    Superposition-free. Per-residue keyed by (ref_chain, ref_resnum)."""
    rc = list(ref)
    best = None
    for perm in itertools.permutations(rc):
        cmap = list(zip(list(pred), perm))
        # require every mapped pair to align
        if any(seq_match(chain_seq(pred[p]), chain_seq(ref[r])) is None for p, r in cmap):
            continue
        Mp, Rp, rid = _atoms_flat({"pred": pred, "ref": ref}, cmap, None, atom_names)
        if len(Rp) < 2:
            continue
        dref = np.sqrt(((Rp[:, None] - Rp[None]) ** 2).sum(-1))
        dmod = np.sqrt(((Mp[:, None] - Mp[None]) ** 2).sum(-1))
        rid_arr = np.array([hash(x) for x in rid])
        iu = np.triu_indices(len(Rp), k=1)
        diff_res = rid_arr[iu[0]] != rid_arr[iu[1]]
        inR0 = (dref[iu] < LDDT_R0) & diff_res
        if inR0.sum() == 0:
            continue
        dd = np.abs(dref[iu][inR0] - dmod[iu][inR0])
        preserved = np.mean([dd < t for t in LDDT_THRESH], axis=0)  # per pair
        g = float(preserved.mean())
        # per-residue: accumulate preserved over pairs touching each residue
        ia, ib = iu[0][inR0], iu[1][inR0]
        acc, cnt = {}, {}
        for k in range(len(preserved)):
            for idx in (ia[k], ib[k]):
                key = rid[idx]
                acc[key] = acc.get(key, 0.0) + preserved[k]
                cnt[key] = cnt.get(key, 0) + 1
        per_res = {key: acc[key] / cnt[key] for key in acc}
        if best is None or g > best[0]:
            best = (g, per_res)
    return best if best else (float("nan"), {})


# ── coverage ────────────────────────────────────────────────────────────────
def coverage(pred, ref, chain_map):
    matched = total = 0
    for p, r in chain_map:
        total += len(ref[r])
        m = seq_match(chain_seq(pred[p]), chain_seq(ref[r]))
        if m is not None:
            matched += len(m[1])
    return matched / total if total else 0.0


# ── driver ──────────────────────────────────────────────────────────────────
def score_one(pred_path, ref_path):
    pred = parse_cif(pred_path)
    ref = parse_cif(ref_path)

    tm, (t, u) = run_usalign(pred_path, ref_path)
    if t is None:
        pred_sup = pred  # no superposition available
    else:
        pred_sup = superpose(pred, t, u)

    cmap = best_chain_map(pred_sup, ref, superposed=(t is not None))

    g_rmsd_aa, pr_rmsd_aa = rmsd_scores(pred_sup, ref, cmap, atom_names=None)
    g_rmsd_bb, pr_rmsd_bb = rmsd_scores(pred_sup, ref, cmap, atom_names=BACKBONE)
    g_lddt_aa, pr_lddt_aa = lddt_scores(pred, ref, atom_names=None)
    g_lddt_bb, pr_lddt_bb = lddt_scores(pred, ref, atom_names=BACKBONE)
    cov = coverage(pred, ref, cmap)

    summary = {
        "tm_score": tm,
        "global_rmsd_allatom": g_rmsd_aa, "global_rmsd_backbone": g_rmsd_bb,
        "global_lddt_allatom": g_lddt_aa, "global_lddt_backbone": g_lddt_bb,
        "coverage": cov,
    }
    # merge per-residue tables keyed by (chain, resnum)
    keys = set(pr_rmsd_aa) | set(pr_lddt_aa)
    per_res = []
    for (ch, num) in sorted(keys):
        per_res.append({
            "chain": ch, "resnum": num,
            "rmsd_allatom": pr_rmsd_aa.get((ch, num)),
            "rmsd_backbone": pr_rmsd_bb.get((ch, num)),
            "lddt_allatom": pr_lddt_aa.get((ch, num)),
            "lddt_backbone": pr_lddt_bb.get((ch, num)),
        })
    return summary, per_res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-id", help="score only this PDB ID")
    args = ap.parse_args()

    summary_rows, perres_rows = [], []
    for variant, (resolve, suffix) in VARIANTS.items():
        try:
            folder = resolve()
        except Exception as exc:
            print(f"  ! {variant}: could not resolve data directory ({exc}), skipping")
            continue
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".cif"):
                continue
            pdb_id = fn.split("_")[0].upper()
            if args.pdb_id and pdb_id != args.pdb_id.strip().upper():
                continue
            ref_path = os.path.join(EXP_DIR, f"{pdb_id}.cif")
            if not os.path.exists(ref_path):
                print(f"  ! no experimental CIF for {pdb_id}, skipping")
                continue
            pred_path = os.path.join(folder, fn)
            summ, per_res = score_one(pred_path, ref_path)
            summ = {"pdb_id": pdb_id, "variant": variant, **summ}
            summary_rows.append(summ)
            for pr in per_res:
                perres_rows.append({"pdb_id": pdb_id, "variant": variant, **pr})
            print(f"  {pdb_id} [{variant}]: TM={summ['tm_score']}, "
                  f"lDDT_aa={summ['global_lddt_allatom']:.3f}, "
                  f"RMSD_bb={summ['global_rmsd_backbone']:.2f} A, cov={summ['coverage']:.2f}")

    if summary_rows:
        sfields = ["pdb_id", "variant", "tm_score",
                   "global_rmsd_allatom", "global_rmsd_backbone",
                   "global_lddt_allatom", "global_lddt_backbone", "coverage"]
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "scores_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sfields)
            w.writeheader(); w.writerows(summary_rows)
        pfields = ["pdb_id", "variant", "chain", "resnum",
                   "rmsd_allatom", "rmsd_backbone", "lddt_allatom", "lddt_backbone"]
        with open(os.path.join(RESULTS, "scores_per_residue.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=pfields)
            w.writeheader(); w.writerows(perres_rows)
        print(f"\nWrote scores_summary.csv ({len(summary_rows)} rows) and "
              f"scores_per_residue.csv ({len(perres_rows)} rows).")


if __name__ == "__main__":
    main()
