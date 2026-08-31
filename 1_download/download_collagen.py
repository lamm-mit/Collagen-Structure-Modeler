"""
download_collagen.py

Downloads clean collagen triple helix structures from the RCSB PDB.

Filtering criteria:
  - Polymer chain instance count is a multiple of 3 (3, 6, 9, 12)
  - All chains are protein-only (no nucleic acids)
  - Each chain 10–200 residues (observed ATOM records)
  - ≥25% glycine content per chain (Gly-X-Y triple helix motif)
  - No internal residue numbering gaps > 5
  - Keywords: "collagen" or "triple helix" in structure metadata

Multi-copy handling:
  Structures with 6/9/12 chains (multiple triple helices in the ASU) are split
  into triple helices by spatial contact (CA-CA proximity), and the first helix
  is kept; additional copies are discarded. Grouping by contact — rather than by
  taking the first three chain IDs — is required because depositors do not always
  label a helix's three chains consecutively (e.g. 8ZMO labels its two helices
  {A,B,F} and {C,D,E}; slicing the first three IDs would mix the two helices).

Exact-duplicate removal:
  Structures whose triplet has an identical multiset of chain sequences are
  collapsed to a single representative, chosen deterministically by:
    1. best (lowest) resolution,
    2. fewest chains in the ASU (prefer a single triple helix),
    3. lexicographically smallest PDB ID.
  This scales automatically as new PDB entries appear.

Outputs:
  - ../0_data/experimental_cif/{PDB_ID}.cif  — one CIF per unique triple helix
  - ../0_data/manifest.csv                    — PDB ID, kind, per-chain sequences
  - rejected.log                          — rejected structures with reasons
"""

import os
import time
import logging
import requests
import gemmi
import numpy as np
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
# Data is written to the shared benchmark dataset folder (../0_data/), relative
# to this script, so the download stage populates the single source of truth.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "0_data")
OUTPUT_DIR = os.path.join(_DATA, "experimental_cif")   # one CIF per structure
_DEPOSITION_KEY = "_pdbx_database_status.recvd_initial_deposition_date"
MANIFEST_OUT = os.path.join(_DATA, "manifest.csv")     # sequences + metadata
REJECTED_LOG = os.path.join(_HERE, "rejected.log")

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL = "https://files.rcsb.org/download/{}.cif"

REQUEST_DELAY = 0.12   # ~8 requests/sec — polite to RCSB
MAX_RETRIES = 3

GLY_FRACTION_MIN = 0.25   # ≥25% Gly → collagen-like (Gly-X-Y ~33%; termini lower it)
MIN_CHAIN_LENGTH = 10     # minimum residues per chain (ATOM)
MAX_CHAIN_LENGTH = 200    # maximum residues per chain (ATOM)
MAX_INTERNAL_GAP = 5      # max allowed consecutive residue number gap within a chain
MAX_CHAIN_MULTIPLES = 4   # accept up to 4× triple helices in ASU (i.e. ≤ 12 chains)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

rejected_fh = open(REJECTED_LOG, "w")


def reject(pdb_id, reason):
    rejected_fh.write(f"{pdb_id}\t{reason}\n")
    rejected_fh.flush()


# ── RCSB helpers ───────────────────────────────────────────────────────────────
def _get(url, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=30, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            log.warning(f"GET {url} failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def search_pdb() -> list[str]:
    """Return PDB IDs matching collagen/triple helix with a multiple-of-3 chain count."""
    allowed_counts = list(range(3, 3 * MAX_CHAIN_MULTIPLES + 1, 3))  # [3, 6, 9, 12]

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # keyword filter
                {
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": [
                        {"type": "terminal", "service": "full_text",
                         "parameters": {"value": "collagen"}},
                        {"type": "terminal", "service": "full_text",
                         "parameters": {"value": "triple helix"}},
                    ],
                },
                # chain count must be one of [3, 6, 9, 12]
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "in",
                        "value": allowed_counts,
                    },
                },
                # protein-only entity types
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True, "results_verbosity": "compact"},
    }

    log.info("Querying RCSB Search API …")
    r = requests.post(SEARCH_URL, json=query, timeout=60)
    r.raise_for_status()
    ids = r.json().get("result_set", [])
    log.info(f"  → {len(ids)} candidates from API search")
    return ids


# ── CIF download ───────────────────────────────────────────────────────────────
def download_cif_raw(pdb_id: str, dest_dir: str) -> str | None:
    """Download the full deposited CIF into a temp location for parsing."""
    path = os.path.join(dest_dir, f"_raw_{pdb_id}.cif")
    if os.path.exists(path):
        return path
    url = DOWNLOAD_URL.format(pdb_id)
    r = _get(url)
    if r is None:
        return None
    with open(path, "wb") as f:
        f.write(r.content)
    time.sleep(REQUEST_DELAY)
    return path


def save_chain_subset(st: gemmi.Structure, chain_names: list[str],
                      out_path: str) -> None:
    """Write a new CIF containing only the specified chains from the first model."""
    st2 = st.clone()
    model = st2[0]
    to_remove = [c.name for c in model if c.name not in chain_names]
    for name in to_remove:
        model.remove_chain(name)
    doc = st2.make_mmcif_document()
    doc.write_file(out_path)


# ── Structure validation ───────────────────────────────────────────────────────
# The 21 residues we can handle. Any other residue in a chain causes rejection.
ALLOWED_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HYP",  # (4R)-hydroxyproline — distinct from PRO
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HYP": "O",  # hydroxyproline — kept distinct, represented as 'O'
}


def residues_to_seq(residues) -> str:
    return "".join(THREE_TO_ONE.get(r.name, "X") for r in residues)


def validate_triplet(chain_group: list[tuple], pdb_id: str) -> list[dict] | None:
    """
    Validate a triplet of (chain_id, aa_residues).
    Returns a list of 3 chain dicts on success, None on failure.
    """
    chain_records = []
    for chain_id, aa_res in chain_group:
        # Check every observed residue is within our 21 allowed types
        unhandled = {r.name for r in aa_res if r.name not in ALLOWED_RESIDUES}
        if unhandled:
            reject(pdb_id,
                   f"chain {chain_id}: contains unhandled residue(s): {', '.join(sorted(unhandled))}")
            return None

        # Sequence from ATOM records — actual observed chemical components
        atom_seq = residues_to_seq(aa_res)
        atom_len = len(atom_seq)

        # Gly-X-Y content — first, so non-collagen proteins are attributed here
        gly_frac = atom_seq.count("G") / atom_len if atom_len > 0 else 0
        if gly_frac < GLY_FRACTION_MIN:
            reject(pdb_id,
                   f"chain {chain_id}: Gly fraction {gly_frac:.2%} < {GLY_FRACTION_MIN:.0%}")
            return None

        # Internal gap check
        seqnums = [r.seqid.num for r in aa_res]
        if len(seqnums) > 1:
            max_gap = max(seqnums[i + 1] - seqnums[i] for i in range(len(seqnums) - 1))
            if max_gap > MAX_INTERNAL_GAP:
                reject(pdb_id,
                       f"chain {chain_id}: internal residue gap {max_gap} > {MAX_INTERNAL_GAP}")
                return None

        # Chain length bounds (based on observed ATOM residues)
        if atom_len < MIN_CHAIN_LENGTH:
            reject(pdb_id, f"chain {chain_id}: length {atom_len} < {MIN_CHAIN_LENGTH}")
            return None
        if atom_len > MAX_CHAIN_LENGTH:
            reject(pdb_id, f"chain {chain_id}: length {atom_len} > {MAX_CHAIN_LENGTH}")
            return None

        chain_records.append({
            "id": chain_id,
            "sequence": atom_seq,   # observed, with HYP as 'O'
            "length": atom_len,
        })

    return chain_records


# ── Triple-helix grouping by spatial contact ─────────────────────────────────
CONTACT_CUTOFF = 5.5  # Å; min CA-CA below this = two chains packed in one helix


def _ca_coords(aa_res) -> np.ndarray:
    """N×3 array of CA coordinates for a chain's residues."""
    pts = []
    for r in aa_res:
        for a in r:
            if a.name == "CA":
                pts.append([a.pos.x, a.pos.y, a.pos.z])
                break
    return np.array(pts) if pts else np.empty((0, 3))


def group_triple_helices(protein_chains: list[tuple]) -> list[list[tuple]]:
    """Group chains into triple helices by CA-CA spatial contact.

    A triple helix is a connected component of chains whose minimum CA-CA
    distance is below CONTACT_CUTOFF. This recovers the true helices even when a
    depositor does not label a helix's three chains with consecutive IDs (e.g.
    8ZMO: helices {A,B,F} and {C,D,E}). Only components of exactly three chains
    are returned, ordered by their earliest member's record position, with chains
    inside each group kept in record order.
    """
    coords = [_ca_coords(aa) for _, aa in protein_chains]
    n = len(protein_chains)

    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if len(coords[i]) and len(coords[j]):
                d = np.min(np.linalg.norm(coords[i][:, None, :] - coords[j][None, :, :], axis=2))
                if d < CONTACT_CUTOFF:
                    adj[i][j] = adj[j][i] = True

    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in range(n):
                if adj[k][j] and not seen[j]:
                    seen[j] = True
                    stack.append(j)
        comps.append(sorted(comp))

    triples = [c for c in sorted(comps, key=lambda c: c[0]) if len(c) == 3]
    return [[protein_chains[k] for k in comp] for comp in triples]


def process_structure(raw_cif: str, pdb_id: str) -> list[dict]:
    """
    Parse a CIF, split into triplets, validate each, deduplicate by sequence.
    Returns a list of passing result dicts (may be empty).
    """
    try:
        st = gemmi.read_structure(raw_cif)
    except Exception as e:
        reject(pdb_id, f"gemmi parse error: {e}")
        return []

    model = st[0]

    # Collect protein chains with CA atoms
    protein_chains = []
    for chain in model:
        aa_res = [r for r in chain if any(a.name == "CA" for a in r)]
        if aa_res:
            protein_chains.append((chain.name, aa_res))

    n = len(protein_chains)
    if n == 0 or n % 3 != 0:
        reject(pdb_id, f"ATOM chain count {n} is not a positive multiple of 3")
        return []

    # Split into triple helices by spatial contact and validate the first helix;
    # ignore any additional copies. Falls back to record order only if no clean
    # 3-chain helix is found (preserves behaviour for simple single-helix cases).
    helices = group_triple_helices(protein_chains)
    triplet = helices[0] if helices else protein_chains[:3]
    chain_records = validate_triplet(triplet, pdb_id)
    if chain_records is None:
        return []

    chain_names = [c["id"] for c in chain_records]
    out_name = f"{pdb_id}.cif"
    save_chain_subset(st, chain_names, os.path.join(OUTPUT_DIR, out_name))

    # Metadata used to pick a single representative among exact-sequence duplicates
    # (see deduplicate()). resolution defaults to 0.0 when unreported by gemmi.
    resolution = getattr(st, "resolution", 0.0) or 0.0
    signature = tuple(sorted(c["sequence"] for c in chain_records))

    deposition_date = (st.info[_DEPOSITION_KEY]
                       if _DEPOSITION_KEY in st.info else "")

    return [{
        "pdb_id": pdb_id,
        "file": out_name,
        "deposition_date": deposition_date,   # PDB initial deposition, YYYY-MM-DD
        "chains": chain_records,
        "signature": signature,   # identical triplet sequences → exact duplicate
        "resolution": resolution,  # Å; lower is better (0.0 = unknown)
        "asu_chains": n,           # protein chains in the ASU (fewer preferred)
    }]


# ── Exact-duplicate removal ──────────────────────────────────────────────────────
def deduplicate(records: list[dict]) -> list[dict]:
    """
    Collapse structures with identical triple-helix sequences to one representative.

    Two structures are exact duplicates when their triplet has the same multiset of
    chain sequences (the `signature`). Among duplicates, the representative is chosen
    by a scalable, deterministic hierarchy:
      1. Best (lowest) resolution — unknown resolution (0.0) always loses.
      2. Fewest chains in the ASU — prefer a single triple helix over packed copies.
      3. Lexicographically smallest PDB ID — final deterministic tie-break.
    Losers are logged to rejected.log and their CIF files removed.
    """
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["signature"]].append(rec)

    def rank(rec: dict):
        res = rec["resolution"] if rec["resolution"] and rec["resolution"] > 0 else float("inf")
        return (res, rec["asu_chains"], rec["pdb_id"])

    kept: list[dict] = []
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        ordered = sorted(group, key=rank)
        winner = ordered[0]
        kept.append(winner)
        for loser in ordered[1:]:
            reject(loser["pdb_id"],
                   f"exact sequence duplicate of {winner['pdb_id']} "
                   f"(kept res={winner['resolution']} Å, asu_chains={winner['asu_chains']})")
            path = os.path.join(OUTPUT_DIR, loser["file"])
            if os.path.exists(path):
                os.remove(path)

    removed = len(records) - len(kept)
    log.info(f"Exact-duplicate removal: {removed} removed, {len(kept)} unique remain")
    # Preserve original discovery order for stable output
    kept_ids = {r["pdb_id"] for r in kept}
    return [r for r in records if r["pdb_id"] in kept_ids]


# ── Excel output ───────────────────────────────────────────────────────────────
def _frame_offset(seq: str) -> int:
    """Reframe offset: the frame (0/1/2) with the most Gly at triplet starts."""
    s = seq.upper()
    return min(range(3), key=lambda k: sum(1 for i in range(k, len(s), 3) if s[i] != "G"))


def write_manifest(records: list[dict], path: str):
    """Write the sequences + metadata in manifest.csv format (the format the
    build/scoring pipeline reads). All fields are derived from the sequences."""
    headers = [
        "pdb_id", "kind", "deposition_date", "n_distinct_chains",
        "gly_start", "frame_offset",
        "has_hyp", "len_a", "len_b", "len_c",
        "chain_a_sequence", "chain_b_sequence", "chain_c_sequence",
    ]
    import csv as _csv
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(headers)
        for rec in records:
            chains = rec["chains"]
            seqs = [c["sequence"] for c in chains]
            n_distinct = len(set(seqs))
            kind = "homotrimer" if n_distinct == 1 else "heterotrimer"
            w.writerow([
                rec["pdb_id"], kind, rec.get("deposition_date", ""), n_distinct,
                "yes" if seqs[0].startswith("G") else "no",
                _frame_offset(seqs[0]),
                "yes" if any("O" in s for s in seqs) else "no",
                chains[0]["length"], chains[1]["length"], chains[2]["length"],
                seqs[0], seqs[1], seqs[2],
            ])
    log.info(f"Saved {len(records)} entries → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdb_ids = search_pdb()

    all_passing = []
    log.info(f"Downloading and validating {len(pdb_ids)} candidates …")

    for pdb_id in tqdm(pdb_ids, unit="structure"):
        raw_cif = download_cif_raw(pdb_id, OUTPUT_DIR)
        if raw_cif is None:
            reject(pdb_id, "download failed")
            continue

        results = process_structure(raw_cif, pdb_id)

        # Remove the raw CIF; keep only the saved chain-subset CIFs
        os.remove(raw_cif)

        all_passing.extend(results)

    # Collapse exact-sequence duplicates to one representative each
    all_passing = deduplicate(all_passing)

    log.info(f"\n{'='*50}")
    log.info(f"Unique triple helices saved: {len(all_passing)}")
    log.info(f"Source PDB entries processed: {len(pdb_ids)}")
    log.info(f"{'='*50}")

    os.makedirs(_DATA, exist_ok=True)
    write_manifest(all_passing, MANIFEST_OUT)
    rejected_fh.close()
    log.info(f"Rejection details → {REJECTED_LOG}")


if __name__ == "__main__":
    main()
