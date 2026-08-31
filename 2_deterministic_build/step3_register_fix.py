#!/usr/bin/env python3
"""
step3_register_fix.py

Correct the inter-chain axial threading of a built triple helix so the three
strands pack with a uniform one-residue stagger and minimal terminal overhang.

The builder always applies a fixed A->B->C stagger. When the three chains differ
in length, the strand that should lead vs trail can be a whole triplet out of
register, so one strand's terminus splays out. This step slides each chain by
whole-triplet steps (which preserve the Gly-X-Y register: G->G, X->X, Y->Y) and
keeps the threading that minimises the axial spread of the chain termini.

The stagger transform is taken from a reference poly-(Gly-Pro-Hyp) homotrimer, so
it is independent of the (possibly wrong) register of the input chains. A shift is
only applied when it reduces overhang by at least MIN_IMPROVEMENT_A, so already
correct structures (and equal-length homotrimers, where overhang gives no signal)
are left untouched.

Public API:
    reregister(chains) -> (chains, (kA, kB, kC), improvement_A)
"""

import argparse
import numpy as np

import step1_backbone_builder as step1

# Only re-thread when overhang drops by at least this much (A). Real fixes give
# >4 A; clean structures and equal-length homotrimers give <0.5 A of noise.
MIN_IMPROVEMENT_A = 2.0

TRIPLET_RANGE = (-2, -1, 0, 1, 2)


# ── rigid-transform helpers (homogeneous 4x4) ───────────────────────────────
def _kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def _homogeneous(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _apply(M, xyz):
    w = M @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    return (w[0], w[1], w[2])


def _transform_chain(chain, M):
    return [{"one_letter": r["one_letter"],
             "atoms": {n: _apply(M, xyz) for n, xyz in r["atoms"].items()}}
            for r in chain]


def _chain_ca(chain):
    return np.array([r["atoms"]["CA"] for r in chain if "CA" in r["atoms"]])


# ── stagger screw (register-independent, from a reference homotrimer) ────────
_REF_SCREW = None


def _stagger_screw():
    global _REF_SCREW
    if _REF_SCREW is None:
        ref = step1.build_backbone(["GPO" * 12], mode="extend")
        a, b = _chain_ca(ref[0]), _chain_ca(ref[1])
        n = min(len(a), len(b))
        _REF_SCREW = _homogeneous(*_kabsch(a[:n], b[:n]))
    return _REF_SCREW


# ── overhang score ──────────────────────────────────────────────────────────
def _helix_axis(chains):
    allca = np.vstack([_chain_ca(c) for c in chains])
    c = allca.mean(0)
    _, _, vt = np.linalg.svd(allca - c)
    return vt[0], c


def _overhang(chains, axis, centroid):
    """Axial spread of the three N-termini plus the three C-termini (lower=better)."""
    def proj(p):
        return float(np.dot(p - centroid, axis))
    n = [proj(_chain_ca(c)[0]) for c in chains]
    cc = [proj(_chain_ca(c)[-1]) for c in chains]
    return (max(n) - min(n)) + (max(cc) - min(cc))


# ── search ──────────────────────────────────────────────────────────────────
def _search(chains):
    S = _stagger_screw()
    Sinv = np.linalg.inv(S)
    axis, centroid = _helix_axis(chains)
    cache = {}

    def power(k):
        if k not in cache:
            out = np.eye(4)
            base = S if k >= 0 else Sinv
            for _ in range(abs(k)):
                out = base @ out
            cache[k] = out
        return cache[k]

    best = None
    for kA in TRIPLET_RANGE:
        for kB in TRIPLET_RANGE:
            for kC in TRIPLET_RANGE:
                trial = [_transform_chain(chains[0], power(3 * kA)),
                         _transform_chain(chains[1], power(3 * kB)),
                         _transform_chain(chains[2], power(3 * kC))]
                sc = _overhang(trial, axis, centroid)
                if best is None or sc < best[0]:
                    best = (sc, (kA, kB, kC), trial)
    sc, (kA, kB, kC), trial = best
    base = min(kA, kB, kC)
    return sc, (kA - base, kB - base, kC - base), trial


def reregister(chains):
    """Re-thread `chains` to minimise terminal overhang. Returns
    (chains, (kA, kB, kC) triplet shifts, improvement_A). Leaves the chains
    unchanged when the best improvement is below MIN_IMPROVEMENT_A."""
    axis, centroid = _helix_axis(chains)
    base = _overhang(chains, axis, centroid)
    best_score, shifts, rethreaded = _search(chains)
    improvement = base - best_score
    if improvement < MIN_IMPROVEMENT_A:
        return chains, (0, 0, 0), improvement
    return rethreaded, shifts, improvement


# ── CLI (backbone PDB in -> re-registered backbone PDB out) ──────────────────
def _main():
    ap = argparse.ArgumentParser(description="Step 3: re-register a triple-helix backbone.")
    ap.add_argument("--backbone", required=True, help="input backbone PDB")
    ap.add_argument("--out", required=True, help="output re-registered backbone PDB")
    args = ap.parse_args()
    import step2_terminal_extension as step2
    chains = step2._read_backbone_pdb(args.backbone)
    chains, shifts, impr = reregister(chains)
    step1.write_backbone_pdb(args.out, chains)
    print(f"wrote {args.out}  shifts A/B/C={shifts}  overhang improvement {impr:.1f} A")


if __name__ == "__main__":
    _main()
