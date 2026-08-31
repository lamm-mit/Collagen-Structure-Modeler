#!/usr/bin/env python3
"""
step2_terminal_extension.py

Phase 1b — terminal overhang completion (the V3 build).

reframe() (in step1) trims each deposited chain to whole Gly-X-Y triplets,
dropping up to 2 residues at the N-terminus and up to 2 at the C-terminus (e.g.
1BKV chain A loses a leading HYP and a trailing GLY). This step puts those
residues back onto the built backbone so the output matches the deposited
residue count.

Because a collagen chain is a regular helix, the rigid-body transform mapping one
residue's backbone (N, CA, C, O) to the next is very nearly constant. We recover
that transform from the two terminal residues by Kabsch superposition and apply
it outward (and its inverse at the N-terminus) to place each appended residue's
backbone atoms. Side chains are added later by step 3 (tleap).

Self-contained: copies the small pieces it needs (reframe logic, residue-name
map) rather than importing step1, so the pipeline has no cross-script coupling.
"""

import argparse
import numpy as np

# One-letter -> PDB comp_id (O = hydroxyproline), matching the THeBuScr port.
THREELET = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "O": "HYP", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR",
    "V": "VAL", "W": "TRP", "Y": "TYR",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def reframe_info(seq: str):
    """(frame_offset f, reframed_seq, n_dropped, c_dropped) — same rule as
    step1.reframe: drop f residues at N-term (fewest non-Gly at triplet starts),
    then drop the trailing partial triplet."""
    s = seq.upper().strip().replace("﻿", "")
    f = min(range(3), key=lambda k: sum(1 for i in range(k, len(s), 3) if s[i] != "G"))
    shifted = s[f:]
    keep = len(shifted) - (len(shifted) % 3)
    return f, shifted[:keep], s[:f], shifted[keep:]


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid transform (R, t) with Q ≈ P @ R.T + t. P, Q are (n, 3) with matched
    rows. No scaling (pure rotation + translation)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def _apply(res: dict, R: np.ndarray, t: np.ndarray, one_letter: str) -> dict:
    """Apply (R, t) to every backbone atom of `res`, returning a new residue with
    identity `one_letter`."""
    atoms = {}
    for name in BACKBONE_ATOMS:
        if name in res["atoms"]:
            v = np.asarray(res["atoms"][name], float)
            atoms[name] = tuple((R @ v + t).tolist())
    return {"one_letter": one_letter, "atoms": atoms}


def _stack(res: dict) -> np.ndarray:
    return np.asarray([res["atoms"][a] for a in BACKBONE_ATOMS], float)


def _extend_one_chain(residues: list, n_dropped: str, c_dropped: str) -> list:
    """Prepend n_dropped residues and append c_dropped residues to one chain.

    Uses the immediately-adjacent terminal residue pair as the per-residue screw.
    (A position-matched transform — sampling a same-junction-type internal pair so
    an appended Gly is placed by a Y->Gly step, etc. — was tried and found to be
    WORSE: the three junction types differ by only ~0.1 A in CA-CA rise, whereas
    sampling the correct type non-locally introduces ~0.8 A error on real
    interrupted helices. Locality dominates, so the local terminal pair wins; any
    residual error is cleaned by the step-5 minimisation.)"""
    if len(residues) < 2:
        return residues  # cannot derive a transform from <2 residues

    out = list(residues)

    # ---- C-terminus: transform mapping residue[-2] -> residue[-1] ----
    if c_dropped:
        R, t = _kabsch(_stack(out[-2]), _stack(out[-1]))
        for letter in c_dropped:                    # N->C order along the chain
            out.append(_apply(out[-1], R, t, letter))

    # ---- N-terminus: transform mapping residue[1] -> residue[0] ----
    if n_dropped:
        R, t = _kabsch(_stack(out[1]), _stack(out[0]))
        # n_dropped are the residues before index 0, in N->C order; place them
        # from the innermost outward, so iterate reversed and insert at front.
        for letter in reversed(n_dropped):
            out.insert(0, _apply(out[0], R, t, letter))

    return out


def extend_termini(chains: list, sequences: list) -> list:
    """Append the reframe-trimmed terminal residues to each built chain.

    chains     : 3 chains (step1 _outhelix format: list of {one_letter, atoms}).
    sequences  : deposited sequence(s) — 1 (homotrimer) or 3 (heterotrimer),
                 same order/expansion used to build the chains.
    Returns 3 extended chains.
    """
    seqs = [s.upper() for s in sequences]
    per_chain_seq = (seqs * 3)[:3] if len(seqs) == 1 else seqs[:3]

    extended = []
    for residues, seq in zip(chains, per_chain_seq):
        _f, _reframed, n_drop, c_drop = reframe_info(seq)
        extended.append(_extend_one_chain(residues, n_drop, c_drop))
    return extended


# ---------------------------------------------------------------------------
# Standalone CLI (backbone PDB in -> extended backbone PDB out)
# ---------------------------------------------------------------------------
def _read_backbone_pdb(path: str) -> list:
    """Parse a Phase-1 backbone PDB into the chains representation."""
    from collections import OrderedDict
    chains_map: "OrderedDict[str, OrderedDict]" = OrderedDict()
    inv = {v: k for k, v in THREELET.items()}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or line[20:22].strip()
            resnum = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            ch = chains_map.setdefault(chain, OrderedDict())
            res = ch.setdefault(resnum, {"one_letter": inv.get(resname, "X"), "atoms": {}})
            res["atoms"][atom] = (x, y, z)
    return [[r for r in ch.values()] for ch in chains_map.values()]


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: append reframe-trimmed terminal residues to a backbone.")
    ap.add_argument("--backbone", required=True, help="input backbone PDB (from step 1)")
    ap.add_argument("--seq", action="append", required=True,
                    help="deposited chain sequence(s); once (homotrimer) or thrice.")
    ap.add_argument("--out", required=True, help="output extended backbone PDB")
    args = ap.parse_args()

    # Local import so the CLI can reuse step1's writer without a package install.
    import step1_backbone_builder as step1
    chains = _read_backbone_pdb(args.backbone)
    chains = extend_termini(chains, args.seq)
    step1.write_backbone_pdb(args.out, chains)
    print(f"wrote {args.out}  (residues/chain={[len(c) for c in chains]})")


if __name__ == "__main__":
    _main()
