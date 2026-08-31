#!/usr/bin/env python3
"""
step5_relax.py — OpenMM energy minimisation (side-chain relaxation).

Takes the tleap-parameterised Amber files that step4 produced ({prefix}.prmtop +
{prefix}.rst7) and runs an energy minimisation to relieve side-chain strain and
clashes, then writes a relaxed heavy-atom mmCIF.

Why reuse the Amber files: hydroxyproline (HYP) is not in OpenMM's standard force
fields, but tleap (ff14SB) already parameterised it into the prmtop. OpenMM reads
the prmtop's parameters directly, so HYP is handled with no extra work.

Design:
  - OBC2 implicit solvent (screens electrostatics so charged side chains don't
    collapse together, as they would in vacuum). OBC2 works with tleap's default
    (mbondi) radii; GBn2 would need a different radii set and rejects HYP atoms.
  - Harmonic positional restraint on backbone heavy atoms (N, CA, C, O) to their
    input positions, so relaxation cleans up side chains without moving the
    backbone away from the target fold.
  - Minimise to OpenMM's default force tolerance, capped at max_iterations.
  - Optional trajectory of the minimisation (captured in chunks) as a compact
    DCD plus a reference PDB, for visualising the relaxation in PyMOL:
      load 8K4X.pdb ; load_traj 8K4X.dcd

Public API:
  relax(prmtop, rst7, out_cif, traj_dcd=None) -> out_cif

Requires OpenMM + parmed (e.g. the MIT_environment conda env).
"""

import os
import numpy as np

import step4_sidechain_builder as step4

BACKBONE = {"N", "CA", "C", "O"}
# Backbone restraint stiffness. 1000 kJ/mol/nm^2 ≈ 2.4 kcal/mol/Å^2 — firm enough
# to hold the fold, soft enough to let the backbone settle slightly.
RESTRAINT_K = 1000.0
MAX_ITERATIONS = 2000     # safety cap on the (single, continuous) minimisation
REPORT_INTERVAL = 5       # save a trajectory frame every N minimiser iterations


def relax(prmtop, rst7, out_cif, traj_dcd=None, restrain_backbone=True):
    """Minimise the Amber system and write a relaxed heavy-atom CIF.

    If `traj_dcd` is given, also write the minimisation trajectory as `traj_dcd`
    (compact binary) plus a sibling reference PDB (same stem, .pdb) that carries
    the topology — load the PDB then the DCD in PyMOL."""
    from openmm import app, unit, LangevinIntegrator, CustomExternalForce
    import parmed

    prm = app.AmberPrmtopFile(prmtop)
    inp = app.AmberInpcrdFile(rst7)
    system = prm.createSystem(implicitSolvent=app.OBC2,
                              nonbondedMethod=app.NoCutoff,
                              constraints=app.HBonds)

    if restrain_backbone:
        restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter(
            "k", RESTRAINT_K * unit.kilojoule_per_mole / unit.nanometer ** 2)
        for p in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(p)
        pos_nm = inp.positions.value_in_unit(unit.nanometer)
        for atom in prm.topology.atoms():
            if atom.name in BACKBONE and (atom.element is None or atom.element.symbol != "H"):
                x, y, z = pos_nm[atom.index]
                restraint.addParticle(atom.index, [x, y, z])
        system.addForce(restraint)

    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                    0.002 * unit.picoseconds)
    sim = app.Simulation(prm.topology, system, integrator)
    sim.context.setPositions(inp.positions)

    # ── minimise as ONE continuous run, sampling frames via a reporter ────────
    # A single minimizeEnergy keeps L-BFGS's curvature history, so it converges
    # deeply (~200 iters for some structures). Chunking into repeated calls would
    # cold-restart L-BFGS each time and stall prematurely — so we capture frames
    # *during* the continuous run with a MinimizationReporter, sampling every
    # REPORT_INTERVAL iterations. Termination is natural (the minimiser stops at
    # the force tolerance), so the frame count scales with how much each structure
    # actually moves.
    dcd_fh = dcd = None
    if traj_dcd:
        os.makedirs(os.path.dirname(os.path.abspath(traj_dcd)), exist_ok=True)
        dcd_fh = open(traj_dcd, "wb")
        dcd = app.DCDFile(dcd_fh, prm.topology, 1.0 * unit.picoseconds)

    if traj_dcd:
        from openmm import MinimizationReporter

        class _FrameReporter(MinimizationReporter):
            def report(self, iteration, x, grad, args):
                if iteration % REPORT_INTERVAL == 0:
                    pos = (np.array(x).reshape(-1, 3)) * unit.nanometer
                    dcd.writeModel(pos)
                return False  # never request an early stop

        sim.minimizeEnergy(maxIterations=MAX_ITERATIONS, reporter=_FrameReporter())
        # final converged frame
        dcd.writeModel(sim.context.getState(getPositions=True).getPositions())
        dcd_fh.close()
    else:
        sim.minimizeEnergy(maxIterations=MAX_ITERATIONS)

    # ── build the minimised ParmEd structure (chains preserved) ───────────────
    final = sim.context.getState(getPositions=True)
    coords = np.array(final.getPositions().value_in_unit(unit.angstrom))
    struct = parmed.load_file(prmtop)
    struct.coordinates = coords

    # Reference PDB (topology for the DCD): write from the MINIMISED coordinates,
    # not the raw tleap output. The pre-minimisation structure contains real
    # steric clashes (independent per-chain side-chain building + re-threading can
    # overlap atoms), and PyMOL infers bonds by distance — so a clashy reference
    # produces spurious cross-chain "bonds". The minimised structure is clash-free,
    # so bonds infer correctly; the DCD then shows those clashes resolving. Chains
    # (A/B/C + TER) are preserved because the prmtop was chain-restored in step4.
    if traj_dcd:
        struct.save(os.path.splitext(traj_dcd)[0] + ".pdb", overwrite=True)

    # ── write relaxed heavy-atom CIF (reuse step4's writer for identical format)
    tmp_rst = out_cif + ".min.rst7"
    struct.save(tmp_rst, overwrite=True, format="rst7")
    try:
        step4.generate_final_structure(prmtop, tmp_rst, out_cif)
    finally:
        if os.path.exists(tmp_rst):
            os.remove(tmp_rst)
    return out_cif


# ── Standalone CLI ───────────────────────────────────────────────────────────
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 5: OpenMM relaxation of a side-chain build.")
    ap.add_argument("--prmtop", required=True, help="Amber topology (from step4 --amber_out)")
    ap.add_argument("--rst7", required=True, help="Amber coordinates")
    ap.add_argument("--out", required=True, help="output relaxed CIF")
    ap.add_argument("--traj", help="optional DCD trajectory path (+ sibling ref PDB)")
    ap.add_argument("--no-restraint", action="store_true", help="free minimisation")
    args = ap.parse_args()
    relax(args.prmtop, args.rst7, args.out, traj_dcd=args.traj,
          restrain_backbone=not args.no_restraint)
    print(f"wrote {args.out}" + (f" (+ trajectory {args.traj})" if args.traj else ""))


if __name__ == "__main__":
    _main()
