#!/usr/bin/env python3
"""
phase2_add_sidechains.py
Add side chains with tleap and restore chain IDs with ParmEd
"""
import collections
import subprocess
import os
import sys
import parmed


def convert_residue_codes(input_pdb, output_pdb):
    """Convert O to HYP while preserving chain IDs"""
    conversions = 0

    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    with open(output_pdb, 'w') as f:
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                resname = line[17:20].strip()
                if resname == 'O':
                    line = line[:17] + 'HYP' + line[20:]
                    conversions += 1
            f.write(line)

    print(f"✓ Converted {conversions} 'O' residues to 'HYP'")
    return conversions


def add_sidechains_tleap(converted_pdb, output_prefix):
    """Use tleap to add side chains (will lose chain IDs temporarily)"""

    tleap_script = f"{output_prefix}_tleap.in"

    with open(tleap_script, 'w') as f:
        f.write(f"""# tleap script for adding side chains
source leaprc.protein.ff14SB
source leaprc.water.tip3p

# Load structure
collagen = loadpdb {converted_pdb}

# Check for errors
check collagen

# Save outputs (chain IDs will be blank - ParmEd fixes this)
savepdb collagen {output_prefix}_nochains.pdb
saveamberparm collagen {output_prefix}_temp.prmtop {output_prefix}_temp.rst7

# Exit
quit
""")

    print(f"✓ Created tleap script: {tleap_script}")

    print("Running tleap to add side chains...")
    result = subprocess.run(
        ['tleap', '-f', tleap_script],
        capture_output=True,
        text=True
    )

    log_file = f"{output_prefix}_tleap.log"
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    if "FATAL" in result.stdout or "Could not" in result.stdout:
        print("ERROR: tleap failed")
        print(result.stdout)
        sys.exit(1)

    for line in result.stdout.split('\n'):
        if "Leap added" in line and "according to residue" in line:
            print(f"  {line.strip()}")
            break

    print(f"✓ tleap completed (chain IDs lost - will be restored)")
    return log_file


def restore_chain_ids_parmed(original_backbone, temp_prmtop, output_prefix):
    """Use ParmEd to restore chain IDs from original backbone"""

    parmed_script = f"{output_prefix}_parmed.in"

    with open(parmed_script, 'w') as f:
        f.write(f"""# ParmEd script to restore chain IDs
addPDB {original_backbone}
outparm {output_prefix}.prmtop
quit
""")

    print("\nRestoring chain IDs with ParmEd...")
    result = subprocess.run(
        ['parmed', temp_prmtop, '-i', parmed_script],
        capture_output=True,
        text=True
    )

    parmed_log = f"{output_prefix}_parmed.log"
    with open(parmed_log, 'w') as f:
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    if result.returncode != 0:
        print("ERROR: ParmEd failed")
        print(result.stdout)
        sys.exit(1)

    print(f"✓ Chain IDs restored from {original_backbone}")
    return parmed_log


def validate_tleap_log(log_path):
    """Gate 2: Scans the tleap log for fatal keywords."""
    if not os.path.exists(log_path):
        return False, "tleap log file not found."

    red_flags = [
        "FATAL",
        "ERROR",
        "Could not find parameter",
        "Unit is not okay",
        "The unperturbed charge of the unit"
    ]

    with open(log_path, 'r') as f:
        for line in f:
            if any(flag in line for flag in red_flags):
                return False, f"tleap Failure Found: {line.strip()}"

    return True, "Success"


# ---------------------------------------------------------------------------
# Residue lookup tables used by fix_cif_for_md
# (mon_nstd_flag, chem_comp type, full name, formula, formula_weight)
# ---------------------------------------------------------------------------
_RESINFO = {
    'GLY': ('y', 'L-peptide linking', 'GLYCINE',          'C2 H5 N O2',      75.032),
    'ALA': ('y', 'L-peptide linking', 'ALANINE',          'C3 H7 N O2',      89.094),
    'VAL': ('y', 'L-peptide linking', 'VALINE',           'C5 H11 N O2',    117.148),
    'LEU': ('y', 'L-peptide linking', 'LEUCINE',          'C6 H13 N O2',    131.175),
    'ILE': ('y', 'L-peptide linking', 'ISOLEUCINE',       'C6 H13 N O2',    131.175),
    'PRO': ('y', 'L-peptide linking', 'PROLINE',          'C5 H9 N O2',     115.131),
    'PHE': ('y', 'L-peptide linking', 'PHENYLALANINE',    'C9 H11 N O2',    165.192),
    'TRP': ('y', 'L-peptide linking', 'TRYPTOPHAN',       'C11 H12 N2 O2',  204.228),
    'MET': ('y', 'L-peptide linking', 'METHIONINE',       'C5 H11 N O2 S',  149.211),
    'SER': ('y', 'L-peptide linking', 'SERINE',           'C3 H7 N O3',     105.093),
    'THR': ('y', 'L-peptide linking', 'THREONINE',        'C4 H9 N O3',     119.120),
    'CYS': ('y', 'L-peptide linking', 'CYSTEINE',         'C3 H7 N O2 S',   121.159),
    'TYR': ('y', 'L-peptide linking', 'TYROSINE',         'C9 H11 N O3',    181.191),
    'HIS': ('y', 'L-peptide linking', 'HISTIDINE',        'C6 H9 N3 O2',    155.157),
    'LYS': ('y', 'L-peptide linking', 'LYSINE',           'C6 H14 N2 O2',   146.190),
    'ARG': ('y', 'L-peptide linking', 'ARGININE',         'C6 H14 N4 O2',   174.204),
    'ASP': ('y', 'L-peptide linking', 'ASPARTIC ACID',    'C4 H7 N O4',     133.104),
    'GLU': ('y', 'L-peptide linking', 'GLUTAMIC ACID',    'C5 H9 N O4',     147.131),
    'ASN': ('y', 'L-peptide linking', 'ASPARAGINE',       'C4 H8 N2 O3',    132.119),
    'GLN': ('y', 'L-peptide linking', 'GLUTAMINE',        'C5 H10 N2 O3',   146.146),
    'HYP': ('n', 'L-peptide linking', '4-HYDROXYPROLINE', 'C5 H9 N O3',    131.130),
}
_1LETTER = {
    'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
    'PRO': 'P', 'PHE': 'F', 'TRP': 'W', 'MET': 'M', 'SER': 'S',
    'THR': 'T', 'CYS': 'C', 'TYR': 'Y', 'HIS': 'H', 'LYS': 'K',
    'ARG': 'R', 'ASP': 'D', 'GLU': 'E', 'ASN': 'N', 'GLN': 'Q',
    'HYP': 'O',
}
_STD_RESIDUES = frozenset(k for k, v in _RESINFO.items() if v[0] == 'y')

# ---------------------------------------------------------------------------
# Chemical component atom and bond tables for non-standard residues.
# These are required by OpenMM's mmCIF parser to form inter-residue peptide
# bonds into non-standard residues like HYP. Without _chem_comp_bond, OpenMM
# cannot identify HYP.N as a linkable peptide nitrogen, causing a
# "missing external bond" error on the preceding residue's C atom.
#
# Data sourced from the wwPDB Chemical Component Dictionary (CCD) entry HYP.
# pdbx_leaving_atom_flag = Y means the atom is only present at chain termini.
# ---------------------------------------------------------------------------
_CHEM_COMP_ATOM = {
    # (atom_id, alt_atom_id, type_symbol, charge, pdbx_leaving_atom_flag)
    'HYP': [
        ('N',   'N',   'N', 0, 'N'),
        ('CA',  'CA',  'C', 0, 'N'),
        ('C',   'C',   'C', 0, 'N'),
        ('O',   'O',   'O', 0, 'N'),
        ('CB',  'CB',  'C', 0, 'N'),
        ('CG',  'CG',  'C', 0, 'N'),
        ('CD',  'CD',  'C', 0, 'N'),
        ('OD1', 'OD1', 'O', 0, 'N'),
        ('OXT', 'OXT', 'O', 0, 'Y'),   # C-terminal only
        ('H',   'H',   'H', 0, 'Y'),   # secondary amine — not present mid-chain
        ('H2',  'H2',  'H', 0, 'Y'),   # N-terminal only
        ('H3',  'H3',  'H', 0, 'Y'),   # N-terminal only
        ('HA',  'HA',  'H', 0, 'N'),
        ('HB2', 'HB2', 'H', 0, 'N'),
        ('HB3', 'HB3', 'H', 0, 'N'),
        ('HG',  'HG',  'H', 0, 'N'),
        ('HD2', 'HD2', 'H', 0, 'N'),
        ('HD3', 'HD3', 'H', 0, 'N'),
        ('HD1', 'HD1', 'H', 0, 'N'),
    ],
}

_CHEM_COMP_BOND = {
    # (atom_id_1, atom_id_2, value_order, pdbx_aromatic_flag, pdbx_stereo_config)
    'HYP': [
        ('N',   'CA',  'SING', 'N', 'N'),
        ('N',   'CD',  'SING', 'N', 'N'),
        ('N',   'H',   'SING', 'N', 'N'),
        ('N',   'H2',  'SING', 'N', 'N'),
        ('N',   'H3',  'SING', 'N', 'N'),
        ('CA',  'C',   'SING', 'N', 'N'),
        ('CA',  'CB',  'SING', 'N', 'N'),
        ('CA',  'HA',  'SING', 'N', 'N'),
        ('C',   'O',   'DOUB', 'N', 'N'),
        ('C',   'OXT', 'SING', 'N', 'N'),
        ('CB',  'CG',  'SING', 'N', 'N'),
        ('CB',  'HB2', 'SING', 'N', 'N'),
        ('CB',  'HB3', 'SING', 'N', 'N'),
        ('CG',  'CD',  'SING', 'N', 'N'),
        ('CG',  'OD1', 'SING', 'N', 'N'),
        ('CG',  'HG',  'SING', 'N', 'N'),
        ('CD',  'HD2', 'SING', 'N', 'N'),
        ('CD',  'HD3', 'SING', 'N', 'N'),
        ('OD1', 'HD1', 'SING', 'N', 'N'),
    ],
}


def _tokenize_cif_line(line):
    """Split a CIF data line into tokens, respecting single- and double-quoted strings."""
    tokens = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch in (' ', '\t'):
            i += 1
        elif ch == '"':
            end = line.find('"', i + 1)
            if end == -1:
                end = len(line) - 1
            tokens.append(line[i:end + 1])
            i = end + 1
        elif ch == "'":
            end = line.find("'", i + 1)
            if end == -1:
                end = len(line) - 1
            tokens.append(line[i:end + 1])
            i = end + 1
        else:
            j = i
            while j < len(line) and line[j] not in (' ', '\t'):
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _is_hydrogen_atom(tokens, type_col=None):
    """Return True when an _atom_site row is a hydrogen/deuterium atom."""
    if type_col is None or type_col >= len(tokens):
        return False
    element = tokens[type_col].strip('"\'').upper()
    return element in {'H', 'D'}


def fix_cif_for_md(cif_path, data_name="structure"):
    """Post-process ParmEd CIF output for full mmCIF compliance.

    Fixes applied:
      1. Data block name: 'data_cell' -> 'data_<data_name>'
      2. Sequential atom IDs: renumber all atoms 1, 2, 3... (fixes -1 placeholders)
      3. ESD columns removed: Cartn_x_esd, Cartn_y_esd, Cartn_z_esd,
         occupancy_esd, B_iso_or_equiv_esd stripped from header and data rows
      4. group_PDB: unquoted; set to HETATM for non-standard residues (e.g. HYP)
      5. label_entity_id: populated from chain->entity mapping (was all '?')
      6. Occupancy: 0.0 -> 1.00 for retained atoms
      7. Metadata blocks added before atom_site:
           _chem_comp         - residue type, formula, standard/modified flag
           _chem_comp_atom    - per-atom element and leaving-atom flags (HYP)
           _chem_comp_bond    - bond topology for HYP (critical for OpenMM)
           _entity            - one entry per unique chain sequence
           _entity_poly       - sequence in one-letter code + canonical form
           _entity_poly_seq   - per-residue sequence table
           _struct_asym       - maps each chain letter to its entity

    Important MD compatibility note:
      Hydrogen rows are stripped from the emitted CIF so downstream tools
      like PDBFixer/OpenMM can rebuild hydrogens using the local HYP
      definitions. This mirrors the working 1BKV -> addHydrogens path.
    """
    ESD_COLS = {
        '_atom_site.Cartn_x_esd',
        '_atom_site.Cartn_y_esd',
        '_atom_site.Cartn_z_esd',
        '_atom_site.occupancy_esd',
        '_atom_site.B_iso_or_equiv_esd',
    }

    with open(cif_path, 'r') as f:
        lines = f.readlines()

    # --- 1. Fix data block name ---
    if lines and lines[0].strip().startswith('data_'):
        lines[0] = f'data_{data_name}\n'

    # --- Locate the _atom_site loop ---
    loop_start = None
    col_names = []
    col_header_linenos = []
    data_start = None

    i = 0
    while i < len(lines):
        if lines[i].strip() == 'loop_':
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                loop_start = i
                i = j
                col_names = []
                col_header_linenos = []
                while i < len(lines) and lines[i].strip().startswith('_atom_site.'):
                    col_names.append(lines[i].strip())
                    col_header_linenos.append(i)
                    i += 1
                data_start = i
                break
        i += 1

    if loop_start is None or not col_names:
        print("  Warning: could not locate _atom_site loop in CIF -- skipping post-processing")
        return

    # Build column index map
    col_idx = {name: ci for ci, name in enumerate(col_names)}
    num_cols = len(col_names)
    esd_indices = sorted(
        [ci for ci, name in enumerate(col_names) if name in ESD_COLS],
        reverse=True,
    )
    id_col     = col_idx.get('_atom_site.id')
    group_col  = col_idx.get('_atom_site.group_PDB')
    comp_col   = col_idx.get('_atom_site.label_comp_id')
    asym_col   = col_idx.get('_atom_site.label_asym_id')
    entity_col = col_idx.get('_atom_site.label_entity_id')
    seq_col    = col_idx.get('_atom_site.label_seq_id')
    occ_col    = col_idx.get('_atom_site.occupancy')
    type_col   = col_idx.get('_atom_site.type_symbol')
    auth_seq_col  = col_idx.get('_atom_site.auth_seq_id')
    auth_comp_col = col_idx.get('_atom_site.auth_comp_id')
    auth_asym_col = col_idx.get('_atom_site.auth_asym_id')
    ins_code_col  = col_idx.get('_atom_site.pdbx_PDB_ins_code')

    # --- First pass: collect per-chain sequences ---
    chain_residues = {}
    unique_restypes = set()
    chain_order = []

    i = data_start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        if stripped.startswith('_') or stripped == 'loop_':
            break
        tokens = _tokenize_cif_line(stripped)
        if len(tokens) >= num_cols:
            chain   = tokens[asym_col].strip('"\'') if asym_col is not None else '?'
            resname = tokens[comp_col].strip('"\'') if comp_col is not None else '?'
            seq_id  = tokens[seq_col].strip('"\'')  if seq_col  is not None else '?'
            if chain not in ('?', '.'):
                if chain not in chain_residues:
                    chain_residues[chain] = {}
                    chain_order.append(chain)
                if seq_id not in ('?', '.'):
                    try:
                        seq_num = int(seq_id)
                        if seq_num not in chain_residues[chain]:
                            auth_seq = (
                                tokens[auth_seq_col].strip('"\'')
                                if auth_seq_col is not None else seq_id
                            )
                            auth_comp = (
                                tokens[auth_comp_col].strip('"\'')
                                if auth_comp_col is not None else resname
                            )
                            auth_asym = (
                                tokens[auth_asym_col].strip('"\'')
                                if auth_asym_col is not None else chain
                            )
                            ins_code = (
                                tokens[ins_code_col].strip('"\'')
                                if ins_code_col is not None else '?'
                            )
                            chain_residues[chain][seq_num] = {
                                'label_comp': resname,
                                'label_seq': seq_num,
                                'auth_seq': auth_seq,
                                'auth_comp': auth_comp,
                                'auth_asym': auth_asym,
                                'ins_code': ins_code,
                            }
                    except ValueError:
                        pass
                if resname not in ('?', '.'):
                    unique_restypes.add(resname)
        i += 1

    chain_sequences = {
        ch: [chain_residues[ch][k]['label_comp'] for k in sorted(chain_residues[ch])]
        for ch in chain_order
    }

    # --- Assign entities (group chains with identical sequences) ---
    entity_seq = {}
    entity_for_chain = {}
    next_eid = 1

    for ch in chain_order:
        seq = chain_sequences[ch]
        matched = next((eid for eid, eseq in entity_seq.items() if eseq == seq), None)
        if matched is None:
            entity_seq[next_eid] = seq
            entity_for_chain[ch] = next_eid
            next_eid += 1
        else:
            entity_for_chain[ch] = matched

    entity_chain_count = collections.Counter(entity_for_chain.values())

    # --- Build metadata blocks to insert before the atom_site loop ---
    def _q(s):
        return f'"{s}"' if ' ' in str(s) else str(s)

    meta = []

    # _chem_comp
    meta += ['#\n', 'loop_\n',
             '_chem_comp.id\n', '_chem_comp.type\n', '_chem_comp.mon_nstd_flag\n',
             '_chem_comp.name\n', '_chem_comp.pdbx_synonyms\n',
             '_chem_comp.formula\n', '_chem_comp.formula_weight\n']
    for resname in sorted(unique_restypes):
        if resname in _RESINFO:
            nstd, rtype, name, formula, fw = _RESINFO[resname]
            meta.append(f'{resname} {_q(rtype)} {nstd} {_q(name)} ? {_q(formula)} {fw}\n')
        else:
            meta.append(f'{resname} "L-peptide linking" . {_q(resname)} ? ? ?\n')

    # _chem_comp_atom -- atom-level detail for non-standard residues.
    # OpenMM uses pdbx_leaving_atom_flag to identify terminal-only atoms
    # (OXT, H2, H3) so it can match mid-chain templates correctly.
    nonstd_with_atoms = [r for r in sorted(unique_restypes) if r in _CHEM_COMP_ATOM]
    if nonstd_with_atoms:
        meta += ['#\n', 'loop_\n',
                 '_chem_comp_atom.comp_id\n', '_chem_comp_atom.atom_id\n',
                 '_chem_comp_atom.alt_atom_id\n', '_chem_comp_atom.type_symbol\n',
                 '_chem_comp_atom.charge\n', '_chem_comp_atom.pdbx_aromatic_flag\n',
                 '_chem_comp_atom.pdbx_leaving_atom_flag\n',
                 '_chem_comp_atom.pdbx_ordinal\n']
        for resname in nonstd_with_atoms:
            for ordinal, (aid, alt, sym, chg, leaving) in enumerate(_CHEM_COMP_ATOM[resname], 1):
                meta.append(f'{resname} {aid} {alt} {sym} {chg} N {leaving} {ordinal}\n')

    # _chem_comp_bond -- bond topology for non-standard residues.
    # Critical: without this OpenMM cannot identify HYP.N as a linkable
    # peptide nitrogen, causing a "missing external bond" error on PRO.C.
    nonstd_with_bonds = [r for r in sorted(unique_restypes) if r in _CHEM_COMP_BOND]
    if nonstd_with_bonds:
        meta += ['#\n', 'loop_\n',
                 '_chem_comp_bond.comp_id\n', '_chem_comp_bond.atom_id_1\n',
                 '_chem_comp_bond.atom_id_2\n', '_chem_comp_bond.value_order\n',
                 '_chem_comp_bond.pdbx_aromatic_flag\n',
                 '_chem_comp_bond.pdbx_stereo_config\n',
                 '_chem_comp_bond.pdbx_ordinal\n']
        for resname in nonstd_with_bonds:
            for ordinal, (a1, a2, order, arom, stereo) in enumerate(_CHEM_COMP_BOND[resname], 1):
                meta.append(f'{resname} {a1} {a2} {order} {arom} {stereo} {ordinal}\n')

    # _entity
    meta += ['#\n', 'loop_\n',
             '_entity.id\n', '_entity.type\n', '_entity.src_method\n',
             '_entity.pdbx_description\n', '_entity.formula_weight\n',
             '_entity.pdbx_number_of_molecules\n', '_entity.details\n']
    for eid in sorted(entity_seq):
        meta.append(f'{eid} polymer man "Collagen chain" ? {entity_chain_count[eid]} ?\n')

    # _entity_poly
    meta += ['#\n', 'loop_\n',
             '_entity_poly.entity_id\n', '_entity_poly.type\n',
             '_entity_poly.nstd_linkage\n', '_entity_poly.nstd_monomer\n',
             '_entity_poly.pdbx_seq_one_letter_code\n',
             '_entity_poly.pdbx_seq_one_letter_code_can\n',
             '_entity_poly.pdbx_strand_id\n']
    for eid in sorted(entity_seq):
        seq = entity_seq[eid]
        has_nonstd = any(r not in _STD_RESIDUES for r in seq)
        olc     = ''.join(_1LETTER.get(r, 'X') for r in seq)
        olc_can = olc.replace('O', 'P')
        nstd_mon = 'y' if has_nonstd else 'n'
        strands  = ','.join(ch for ch in chain_order if entity_for_chain[ch] == eid)
        meta.append(f'{eid} polypeptide(L) no {nstd_mon} {olc} {olc_can} {strands}\n')

    # _entity_poly_seq
    meta += ['#\n', 'loop_\n',
             '_entity_poly_seq.entity_id\n', '_entity_poly_seq.num\n',
             '_entity_poly_seq.mon_id\n', '_entity_poly_seq.hetero\n']
    for eid in sorted(entity_seq):
        for pos, resname in enumerate(entity_seq[eid], 1):
            meta.append(f'{eid} {pos} {resname} n\n')

    # _struct_asym
    meta += ['#\n', 'loop_\n',
             '_struct_asym.id\n', '_struct_asym.pdbx_blank_PDB_chainid_flag\n',
             '_struct_asym.pdbx_modified\n', '_struct_asym.entity_id\n',
             '_struct_asym.details\n']
    for ch in chain_order:
        meta.append(f'{ch} N N {entity_for_chain[ch]} ?\n')

    # _struct_conn -- explicit peptide connectivity between adjacent residues.
    # This mirrors the key metadata present in 1BKV.cif and helps OpenMM
    # distinguish internal residues from N-/C-terminal ones in generated CIFs.
    conn_rows = []
    conn_id = 1
    for ch in chain_order:
        seq_nums = sorted(chain_residues[ch])
        for prev_seq, next_seq in zip(seq_nums, seq_nums[1:]):
            if next_seq != prev_seq + 1:
                continue
            prev_res = chain_residues[ch][prev_seq]
            next_res = chain_residues[ch][next_seq]
            row_tokens = [
                f"covale{conn_id}",
                "covale",
                "both",
                "?",
                ch,
                prev_res['label_comp'],
                str(prev_res['label_seq']),
                "C",
                "?",
                prev_res['ins_code'] if prev_res['ins_code'] not in ('', '.') else "?",
                "?",
                "1_555",
                ch,
                next_res['label_comp'],
                str(next_res['label_seq']),
                "N",
                "?",
                next_res['ins_code'] if next_res['ins_code'] not in ('', '.') else "?",
                prev_res['auth_asym'],
                prev_res['auth_comp'],
                str(prev_res['auth_seq']),
                next_res['auth_asym'],
                next_res['auth_comp'],
                str(next_res['auth_seq']),
                "1_555",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
            ]
            conn_rows.append(" ".join(row_tokens) + "\n")
            conn_id += 1

    if conn_rows:
        meta += ['#\n', 'loop_\n',
                 '_struct_conn.id\n',
                 '_struct_conn.conn_type_id\n',
                 '_struct_conn.pdbx_leaving_atom_flag\n',
                 '_struct_conn.pdbx_PDB_id\n',
                 '_struct_conn.ptnr1_label_asym_id\n',
                 '_struct_conn.ptnr1_label_comp_id\n',
                 '_struct_conn.ptnr1_label_seq_id\n',
                 '_struct_conn.ptnr1_label_atom_id\n',
                 '_struct_conn.pdbx_ptnr1_label_alt_id\n',
                 '_struct_conn.pdbx_ptnr1_PDB_ins_code\n',
                 '_struct_conn.pdbx_ptnr1_standard_comp_id\n',
                 '_struct_conn.ptnr1_symmetry\n',
                 '_struct_conn.ptnr2_label_asym_id\n',
                 '_struct_conn.ptnr2_label_comp_id\n',
                 '_struct_conn.ptnr2_label_seq_id\n',
                 '_struct_conn.ptnr2_label_atom_id\n',
                 '_struct_conn.pdbx_ptnr2_label_alt_id\n',
                 '_struct_conn.pdbx_ptnr2_PDB_ins_code\n',
                 '_struct_conn.ptnr1_auth_asym_id\n',
                 '_struct_conn.ptnr1_auth_comp_id\n',
                 '_struct_conn.ptnr1_auth_seq_id\n',
                 '_struct_conn.ptnr2_auth_asym_id\n',
                 '_struct_conn.ptnr2_auth_comp_id\n',
                 '_struct_conn.ptnr2_auth_seq_id\n',
                 '_struct_conn.ptnr2_symmetry\n',
                 '_struct_conn.pdbx_ptnr3_label_atom_id\n',
                 '_struct_conn.pdbx_ptnr3_label_seq_id\n',
                 '_struct_conn.pdbx_ptnr3_label_comp_id\n',
                 '_struct_conn.pdbx_ptnr3_label_asym_id\n',
                 '_struct_conn.pdbx_ptnr3_label_alt_id\n',
                 '_struct_conn.pdbx_ptnr3_PDB_ins_code\n',
                 '_struct_conn.details\n',
                 '_struct_conn.pdbx_dist_value\n',
                 '_struct_conn.pdbx_value_order\n',
                 '_struct_conn.pdbx_role\n']
        meta.extend(conn_rows)
        meta += ['#\n',
                 '_struct_conn_type.id covale\n',
                 '_struct_conn_type.criteria ?\n',
                 '_struct_conn_type.reference ?\n']

    meta.append('#\n')

    # --- Remove ESD column header lines ---
    for ci in esd_indices:
        lines[col_header_linenos[ci]] = ''

    # --- Second pass: fix atom data rows ---
    atom_serial = 0
    i = data_start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        if stripped.startswith('_') or stripped == 'loop_':
            break
        tokens = _tokenize_cif_line(stripped)
        if len(tokens) < num_cols:
            i += 1
            continue

        if _is_hydrogen_atom(tokens, type_col=type_col):
            lines[i] = ''
            i += 1
            continue

        # 2. Sequential atom ID
        atom_serial += 1
        if id_col is not None:
            tokens[id_col] = str(atom_serial)

        # 4. group_PDB -- unquote; HETATM for non-standard residues
        if group_col is not None and comp_col is not None:
            resname = tokens[comp_col].strip('"\'')
            tokens[group_col] = 'ATOM' if resname in _STD_RESIDUES else 'HETATM'

        # 5. label_entity_id
        if entity_col is not None and asym_col is not None:
            ch = tokens[asym_col].strip('"\'')
            tokens[entity_col] = str(entity_for_chain.get(ch, '?'))

        # 6. Occupancy: 0.0 -> 1.00
        if occ_col is not None:
            try:
                if float(tokens[occ_col]) == 0.0:
                    tokens[occ_col] = '1.00'
            except (ValueError, IndexError):
                pass

        # 3. Remove ESD columns (reverse order keeps earlier indices valid)
        for ci in esd_indices:
            if ci < len(tokens):
                tokens.pop(ci)

        lines[i] = ' '.join(tokens) + '\n'
        i += 1

    # --- Insert metadata before the atom_site loop ---
    lines[loop_start:loop_start] = meta

    with open(cif_path, 'w') as f:
        f.writelines(lines)

    print(f"  Fixed CIF: {atom_serial} atoms, {len(entity_seq)} entity/entities "
          f"across {len(chain_order)} chains; _chem_comp_bond added for "
          f"{nonstd_with_bonds}; data block 'data_{data_name}'")


def validate_md_ready_cif(cif_path):
    """Check that the final CIF is compatible with the MD preparation flow."""
    with open(cif_path, 'r') as f:
        lines = f.readlines()

    atom_rows = 0
    hydrogen_rows = 0
    bad_hyp_names = set()
    in_atom_site = False
    headers = []
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'loop_':
            in_atom_site = False
            headers = []
            data_start = None
            continue
        if stripped.startswith('_atom_site.'):
            headers.append(stripped)
            in_atom_site = True
            data_start = i + 1
            continue
        if stripped.startswith('_'):
            in_atom_site = False
            continue
        if not in_atom_site or not stripped or stripped.startswith('#'):
            continue

        tokens = _tokenize_cif_line(stripped)
        if len(tokens) < len(headers):
            continue

        col_idx = {name: ci for ci, name in enumerate(headers)}
        type_col = col_idx.get('_atom_site.type_symbol')
        comp_col = col_idx.get('_atom_site.label_comp_id')
        atom_col = col_idx.get('_atom_site.label_atom_id')

        atom_rows += 1
        if _is_hydrogen_atom(tokens, type_col=type_col):
            hydrogen_rows += 1
        if comp_col is not None and atom_col is not None:
            if tokens[comp_col].strip('"\'') == 'HYP':
                atom_name = tokens[atom_col].strip('"\'')
                if atom_name in {'HD22', 'HD23', 'HXT'}:
                    bad_hyp_names.add(atom_name)

    issues = []
    if hydrogen_rows:
        issues.append(f"found {hydrogen_rows} hydrogen/deuterium atom_site rows")
    if bad_hyp_names:
        issues.append(f"found incompatible HYP atom names: {sorted(bad_hyp_names)}")

    if issues:
        raise RuntimeError(
            f"Generated CIF is not MD-ready ({', '.join(issues)}). "
            "Expected a heavy-atom CIF so PDBFixer/OpenMM can rebuild hydrogens."
        )

    print(f"  MD-ready CIF validated: {atom_rows} heavy atoms, no atom_site hydrogens present.")


def generate_final_structure(prmtop, rst7, output_cif, output_pdb=None):
    """Generate final CIF (and optionally PDB) using ParmEd Python API."""

    print("Generating final structure with ParmEd...")

    structure = parmed.load_file(prmtop, xyz=rst7)

    # primary output: mmCIF
    structure.write_cif(output_cif, renumber=False)
    data_name = os.path.splitext(os.path.basename(output_cif))[0]
    fix_cif_for_md(output_cif, data_name=data_name)
    validate_md_ready_cif(output_cif)
    print(f"Final CIF created: {output_cif}")

    # secondary output: PDB (backward compatibility + chain verification)
    if output_pdb:
        structure.write_pdb(output_pdb, renumber=False)
        print(f"Final PDB created: {output_pdb}")


def verify_chains(pdb_file):
    """Verify 3 chains with IDs present"""
    chains = set()
    chain_residues = {}

    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                resnum = line[22:26].strip()
                if chain.strip():
                    chains.add(chain)
                    if chain not in chain_residues:
                        chain_residues[chain] = set()
                    chain_residues[chain].add(resnum)

    print(f"\n✓ Chain verification:")
    print(f"  Chains found: {sorted(chains)}")

    for chain in sorted(chains):
        print(f"  Chain {chain}: {len(chain_residues[chain])} residues")

    if len(chains) == 3 and chains == {'A', 'B', 'C'}:
        print(f"  ✓✓✓ PERFECT: 3 chains (A, B, C) preserved!")
        return True
    else:
        print(f"  ✗ WARNING: Expected 3 chains (A, B, C)")
        return False


def add_sidechains(backbone_pdb, out_cif, amber_out=None):
    """Add side chains to a Phase-1 backbone PDB and write the final mmCIF.

    Runs the same 4-step tleap/ParmEd flow as the original phase2 pipeline, in a
    temp workdir, then copies the final CIF to `out_cif`. Requires `tleap` on
    PATH and `parmed` importable (e.g. the MIT_environment conda env).

    If `amber_out` (a path prefix) is given, the tleap-parameterised Amber files
    are copied out as `{amber_out}.prmtop` and `{amber_out}.rst7` so a downstream
    step (step5 OpenMM relaxation) can reuse the ff14SB parameters — this is how
    hydroxyproline stays parameterised without OpenMM needing to know HYP.
    """
    import shutil
    import tempfile

    work = tempfile.mkdtemp(prefix="sidechain_")
    try:
        prefix = os.path.join(work, "structure")
        converted = f"{prefix}_converted.pdb"
        convert_residue_codes(backbone_pdb, converted)
        add_sidechains_tleap(converted, prefix)
        restore_chain_ids_parmed(backbone_pdb, f"{prefix}_temp.prmtop", prefix)
        final_cif = f"{prefix}_final.cif"
        generate_final_structure(f"{prefix}.prmtop", f"{prefix}_temp.rst7", final_cif)
        os.makedirs(os.path.dirname(os.path.abspath(out_cif)), exist_ok=True)
        shutil.copyfile(final_cif, out_cif)
        if amber_out is not None:
            os.makedirs(os.path.dirname(os.path.abspath(amber_out)), exist_ok=True)
            shutil.copyfile(f"{prefix}.prmtop", f"{amber_out}.prmtop")
            shutil.copyfile(f"{prefix}_temp.rst7", f"{amber_out}.rst7")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return out_cif


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python phase2_add_sidechains.py backbone.pdb output_prefix")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_prefix = sys.argv[2]

    cleanup_files = [
        f"{output_prefix}_converted.pdb",
        f"{output_prefix}_nochains.pdb",
        f"{output_prefix}_temp.prmtop",
        f"{output_prefix}_temp.rst7",
        f"{output_prefix}.prmtop",
        f"{output_prefix}_complete.pdb",
        f"{output_prefix}_tleap.in",
        f"{output_prefix}_tleap.log",
        f"{output_prefix}_parmed.in",
        f"{output_prefix}_parmed.log"
    ]

    for f in cleanup_files:
        if os.path.exists(f):
            os.remove(f)

    print("=" * 60)
    print("PHASE 2: Side Chain Addition + Chain ID Restoration")
    print("=" * 60)

    print("\nStep 1: Converting O -> HYP...")
    converted_pdb = f"{output_prefix}_converted.pdb"
    convert_residue_codes(input_pdb, converted_pdb)

    print("\nStep 2: Adding side chains with tleap...")
    tleap_log = add_sidechains_tleap(converted_pdb, output_prefix)

    log_file = f"{output_prefix}_tleap.log"
    is_ok, error_msg = validate_tleap_log(log_file)
    if not is_ok:
        print(f"\n[GATE 2 FAILED]: {error_msg}")
        print("Stopping pipeline: The all-atom structure is physically invalid.")
        sys.exit(1)

    print("\nStep 3: Restoring chain IDs with ParmEd...")
    parmed_log = restore_chain_ids_parmed(
        input_pdb,
        f"{output_prefix}_temp.prmtop",
        output_prefix
    )

    print("\nStep 4: Generating final structure...")
    generate_final_structure(
        f"{output_prefix}.prmtop",
        f"{output_prefix}_temp.rst7",
        f"{output_prefix}_complete.cif",
        f"{output_prefix}_complete.pdb"
    )

    print("\nStep 5: Verifying structure...")
    verify_chains(f"{output_prefix}_complete.pdb")

    print("\n" + "=" * 60)
    print("✓ Phase 2 complete")
    print("=" * 60)
    print(f"  {output_prefix}_complete.cif  <- FINAL STRUCTURE (mmCIF)")
    print(f"  {output_prefix}_complete.pdb  <- FINAL STRUCTURE (PDB)")
    print(f"  {output_prefix}.prmtop        <- AMBER topology")
    print(f"  {output_prefix}_temp.rst7     <- AMBER coordinates")
    print(f"  {tleap_log}                   <- tleap log")
    print(f"  {parmed_log}                  <- ParmEd log")
