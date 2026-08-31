#!/usr/bin/env python3
"""
targets.py — model-agnostic loader for the prediction targets.

Reads the benchmark manifest (currently ../../0_data/manifest.csv; later the
HuggingFace parquet — change only `load_targets`) and yields one Target per PDB
entry with its deposited chain sequence(s). Each ML model driver (Boltz, AF3,
Chai, …) consumes these Targets and encodes them into its own input format.

Sequences use one-letter codes with `O` = hydroxyproline (HYP), exactly as in the
manifest and the experimental structures.
"""

import csv
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from data_locations import manifest_path as _manifest_path  # noqa: E402


@dataclass(frozen=True)
class Target:
    pdb_id: str
    kind: str                 # "homotrimer" | "heterotrimer"
    sequences: tuple          # 1 sequence (homotrimer) or 3 (heterotrimer)

    @property
    def is_homotrimer(self) -> bool:
        return self.kind == "homotrimer"

    def chain_sequences(self) -> list:
        """The three physical chain sequences (homotrimer sequence built x3)."""
        s = list(self.sequences)
        return s * 3 if len(s) == 1 else s[:3]


def _read_rows(path: str):
    """Manifest rows as dicts, from either .csv or .parquet."""
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_targets(manifest: str = None) -> list:
    """Return [Target, ...] from the manifest. Single source of sequence truth.

    `manifest` defaults to whatever data_locations resolves — the working
    tree locally, or the HuggingFace copy elsewhere. Pass a path to override.
    """
    targets = []
    for row in _read_rows(manifest or _manifest_path("csv")):
        kind = str(row["kind"]).strip().lower()
        if kind == "homotrimer":
            seqs = (str(row["chain_a_sequence"]).strip().upper(),)
        else:
            seqs = (str(row["chain_a_sequence"]).strip().upper(),
                    str(row["chain_b_sequence"]).strip().upper(),
                    str(row["chain_c_sequence"]).strip().upper())
        targets.append(Target(str(row["pdb_id"]).strip().upper(), kind, seqs))
    return targets


if __name__ == "__main__":
    ts = load_targets()
    homo = sum(t.is_homotrimer for t in ts)
    print(f"{len(ts)} targets  ({homo} homotrimer, {len(ts) - homo} heterotrimer)")
    for t in ts[:3]:
        print(f"  {t.pdb_id} [{t.kind}] chains={[len(s) for s in t.chain_sequences()]}")
