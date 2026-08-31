#!/usr/bin/env python3
"""
step5_anneal.py — simulated-annealing side-chain relaxation.

A drop-in ALTERNATIVE to step5_relax.relax: same signature, same inputs (the
tleap Amber prmtop + rst7), same heavy-atom CIF output. The difference is the
method:

  step5_relax   : energy minimisation only — settles side chains into the LOCAL
                  minimum of whatever rotamer tleap placed them in (cannot cross
                  barriers, so it cannot change rotamer or Pro/Hyp ring pucker).
  step5_anneal  : heat → cool → minimise (backbone restrained). The high-T MD
                  lets side chains cross barriers and sample OTHER rotamers and
                  Pro/Hyp puckers, then cooling+minimisation settles them. Runs a
                  few seeds and keeps the lowest-energy result.

Rationale: the per-atom lDDT diagnosis showed our accuracy gap to Boltz is mostly
in side chains that minimisation can't reprovision (esp. interruption-region
residues). Annealing samples all side chains uniformly — including Pro/Hyp rings —
using the force field we already have (HYP parameterised by tleap), with no
rotamer-library / SCWRL dependency.

Backbone is position-restrained throughout, so the fold is preserved and only the
side chains (and ring puckers) are free to reorganise.

Public API (identical to step5_relax):
  anneal(prmtop, rst7, out_cif, traj_dcd=None, restrain_backbone=True) -> out_cif

Requires OpenMM + parmed (e.g. the MIT_environment conda env).
"""

import os
import numpy as np

import step4_sidechain_builder as step4

BACKBONE = {"N", "CA", "C", "O"}
RESTRAINT_K = 1000.0        # kJ/mol/nm^2 on backbone heavy atoms (as in step5_relax)

# Annealing schedule (tunable). A few short heat→cool cycles from different seeds;
# keep the lowest-energy structure. Kept modest so 75 structures run in minutes.
N_SEEDS = 3
T_SCHEDULE = [600.0, 450.0, 300.0, 150.0]   # K, cooling stages
EQUIL_STEPS = 250            # MD steps at the top temperature before cooling
STEPS_PER_STAGE = 250        # MD steps per cooling stage (2 fs → 0.5 ps each)
TIMESTEP_PS = 0.002
FINAL_MAX_ITERATIONS = 2000  # minimisation after each anneal
TRAJ_STRIDE = 25             # save a trajectory frame every N MD steps (if traj_dcd)


def anneal(prmtop, rst7, out_cif, traj_dcd=None, restrain_backbone=True):
    """Simulated-annealing side-chain relaxation; writes a relaxed heavy-atom CIF.

    If `traj_dcd` is given, writes the MD sampling as a DCD (+ sibling reference
    PDB from the final structure) for visualisation, same convention as step5_relax."""
    from openmm import (app, unit, LangevinMiddleIntegrator, CustomExternalForce)
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

    integrator = LangevinMiddleIntegrator(T_SCHEDULE[0] * unit.kelvin,
                                          1 / unit.picosecond,
                                          TIMESTEP_PS * unit.picoseconds)
    sim = app.Simulation(prm.topology, system, integrator)

    # optional trajectory of the MD sampling
    dcd_fh = dcd = None
    if traj_dcd:
        os.makedirs(os.path.dirname(os.path.abspath(traj_dcd)), exist_ok=True)
        dcd_fh = open(traj_dcd, "wb")
        dcd = app.DCDFile(dcd_fh, prm.topology, 1.0 * unit.picoseconds)

    def run_md(nsteps, temperature):
        integrator.setTemperature(temperature * unit.kelvin)
        if dcd is None:
            sim.step(nsteps)
        else:
            done = 0
            while done < nsteps:
                chunk = min(TRAJ_STRIDE, nsteps - done)
                sim.step(chunk); done += chunk
                dcd.writeModel(sim.context.getState(getPositions=True).getPositions())

    best = None  # (energy_kj, positions_angstrom)
    for seed in range(N_SEEDS):
        sim.context.setPositions(inp.positions)
        sim.minimizeEnergy(maxIterations=200)                 # relieve clashes first
        sim.context.setVelocitiesToTemperature(T_SCHEDULE[0] * unit.kelvin, seed + 1)
        run_md(EQUIL_STEPS, T_SCHEDULE[0])                    # equilibrate hot
        for temp in T_SCHEDULE:                               # cool in stages
            run_md(STEPS_PER_STAGE, temp)
        sim.minimizeEnergy(maxIterations=FINAL_MAX_ITERATIONS)
        state = sim.context.getState(getPositions=True, getEnergy=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if best is None or e < best[0]:
            best = (e, np.array(state.getPositions().value_in_unit(unit.angstrom)))

    if dcd_fh:
        dcd_fh.close()

    # write the lowest-energy structure (reuse step4's CIF writer / chain restore)
    struct = parmed.load_file(prmtop)
    struct.coordinates = best[1]
    if traj_dcd:
        struct.save(os.path.splitext(traj_dcd)[0] + ".pdb", overwrite=True)  # ref topology
    tmp_rst = out_cif + ".anneal.rst7"
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
    ap = argparse.ArgumentParser(description="Step 5 (alt): simulated-annealing relaxation.")
    ap.add_argument("--prmtop", required=True)
    ap.add_argument("--rst7", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", help="optional DCD trajectory (+ sibling ref PDB)")
    ap.add_argument("--no-restraint", action="store_true")
    args = ap.parse_args()
    anneal(args.prmtop, args.rst7, args.out, traj_dcd=args.traj,
           restrain_backbone=not args.no_restraint)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
