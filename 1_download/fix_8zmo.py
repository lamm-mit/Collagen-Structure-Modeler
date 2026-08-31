"""One-off: regenerate ONLY 8ZMO with the corrected contact-based helix grouping.

Does not re-query the PDB. Downloads 8ZMO, reruns process_structure (now grouping
by spatial contact), overwrites 0_data/experimental_cif/8ZMO.cif, and updates the
single 8ZMO row in 0_data/manifest.csv. rejected.log is preserved.
"""
import os
import csv
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REJ = os.path.join(HERE, "rejected.log")
BAK = REJ + ".preserve"

# Preserve rejected.log — importing the module truncates it (open "w" at import).
if os.path.exists(REJ):
    shutil.copy(REJ, BAK)

import download_collagen as dc

PID = "8ZMO"
raw = dc.download_cif_raw(PID, dc.OUTPUT_DIR)
assert raw, "download failed"
results = dc.process_structure(raw, PID)
os.remove(raw)
dc.rejected_fh.close()

# Restore rejected.log
if os.path.exists(BAK):
    shutil.move(BAK, REJ)

assert results, "8ZMO did not pass — unexpected"
rec = results[0]
chains = rec["chains"]
seqs = [c["sequence"] for c in chains]
print(f"{PID} corrected helix chains:")
for c in chains:
    print(f"  chain {c['id']}: len {c['length']}  {c['sequence']}")

# Build the corrected manifest row (same derivations as write_manifest)
n_distinct = len(set(seqs))
new_row = {
    "pdb_id": PID,
    "kind": "homotrimer" if n_distinct == 1 else "heterotrimer",
    "n_distinct_chains": n_distinct,
    "gly_start": "yes" if seqs[0].startswith("G") else "no",
    "frame_offset": dc._frame_offset(seqs[0]),
    "has_hyp": "yes" if any("O" in s for s in seqs) else "no",
    "len_a": chains[0]["length"], "len_b": chains[1]["length"], "len_c": chains[2]["length"],
    "chain_a_sequence": seqs[0], "chain_b_sequence": seqs[1], "chain_c_sequence": seqs[2],
}

# Replace only the 8ZMO row, preserving order and all other rows.
rows = list(csv.DictReader(open(dc.MANIFEST_OUT)))
fieldnames = list(rows[0].keys())
replaced = False
for i, r in enumerate(rows):
    if r["pdb_id"].upper() == PID:
        old = dict(r)
        # keep columns fix_8zmo does not derive (e.g. deposition_date)
        rows[i] = {k: str(new_row[k]) if k in new_row else r[k] for k in fieldnames}
        replaced = True
        break
assert replaced, "8ZMO row not found in manifest"

with open(dc.MANIFEST_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("\nmanifest row updated:")
print("  old:", {k: old[k] for k in ("len_a", "len_b", "len_c", "chain_c_sequence")})
print("  new:", {k: rows[i][k] for k in ("len_a", "len_b", "len_c", "chain_c_sequence")})
print("Done.")
