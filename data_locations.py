#!/usr/bin/env python3
"""
data_locations.py — single point of truth for where benchmark data lives.

There are two layouts for the same data:

  working tree            HuggingFace dataset
  0_data/experimental_cif      experimental/experimental_cif
  2_deterministic_build/…      cdsm/<stage>
  3_ml_prediction/outputs/…    predictions/<model>
  4_scoring/results            scores

LAYOUT below maps between them, and drives both directions: reads resolve
through it here, and upload_to_huggingface.py walks the same table to push.
Add a dataset section in one place, not two.

Resolution order for reads:
  1. COLLAGEN_DATA_ROOT, if set — a local copy in HuggingFace layout.
  2. The working tree, if the legacy directory exists (your machine, today).
  3. HuggingFace, downloaded lazily per-prefix into the local hub cache.

All three return an ordinary directory path, so downstream os.path.join and
gemmi.read_structure calls need no changes.

Writes never go through here. Producer scripts write into the working tree via
local_out(); uploading stays a deliberate act, never a side effect of a run.

Usage:
    from data_locations import experimental_cif_dir, predictions_dir
    exp = experimental_cif_dir()
    af3 = predictions_dir("af3_msa")

    COLLAGEN_DATA_ROOT=~/collagen_data python score.py   # explicit local copy
    COLLAGEN_FORCE_HUB=1 python score.py                 # ignore working tree
"""

import functools
import os

REPO_ID = "CollagenHelixLabs/cdsm_benchmarking_data"
REPO_TYPE = "dataset"

_HERE = os.path.dirname(os.path.abspath(__file__))

# AlphaFold3 appears twice, always named for its MSA condition: there is no
# bare "af3", so no label can silently mean one thing here and another in an
# older results table.
MODELS = ("boltz", "chai", "protenix", "af3_msa", "af3_nomsa")

CDSM_STAGES = (
    "coreonly",
    "fullseq",
    "fullseq_reregistered",
    "fullseq_reregistered_relaxed",
    "fullseq_reregistered_annealed",
)


def repo_root() -> str:
    """The git working tree. Used for writes and for tools/, never for data."""
    return _HERE


# ── layout table: hf_prefix -> path in the working tree ──────────────────────
def _build_layout() -> dict:
    layout = {
        "experimental": os.path.join("0_data",),
        "scores": os.path.join("4_scoring", "results"),
    }
    for stage in CDSM_STAGES:
        layout[f"cdsm/{stage}"] = os.path.join(
            "2_deterministic_build", "outputs", f"gen_struct_{stage}")
    for model in MODELS:
        layout[f"predictions/{model}"] = os.path.join(
            "3_ml_prediction", "outputs", model)
    return layout


LAYOUT = _build_layout()


# ── resolution ───────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _hub_snapshot(patterns: tuple, ignore: tuple = ()) -> str:
    """Download the given prefixes from HuggingFace; return the snapshot dir."""
    from huggingface_hub import snapshot_download  # imported lazily

    return snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=list(patterns) or None,
        ignore_patterns=list(ignore) or None,
    )


def data_root(prefix: str, patterns: tuple = None, ignore: tuple = ()) -> str:
    """Resolve one dataset prefix (e.g. "predictions/af3_msa") to a real directory.

    Prefers an explicit COLLAGEN_DATA_ROOT, then the working tree, then the Hub.
    Set COLLAGEN_FORCE_HUB=1 to skip the working tree and verify that what was
    uploaded actually reproduces your local results.

    `patterns` narrows what is fetched from the Hub — pass e.g.
    ("experimental/manifest.csv",) to read the manifest without pulling the
    80 experimental structures alongside it. Ignored for local resolution.
    """
    if prefix not in LAYOUT:
        raise ValueError(f"unknown dataset prefix {prefix!r}; "
                         f"expected one of {sorted(LAYOUT)}")

    explicit = os.environ.get("COLLAGEN_DATA_ROOT")
    if explicit:
        return os.path.join(os.path.abspath(os.path.expanduser(explicit)), prefix)

    if not os.environ.get("COLLAGEN_FORCE_HUB"):
        local = os.path.join(repo_root(), LAYOUT[prefix])
        if os.path.isdir(local):
            return local

    return os.path.join(_hub_snapshot(patterns or (f"{prefix}/**",), ignore), prefix)


# ── read accessors ───────────────────────────────────────────────────────────
def experimental_cif_dir() -> str:
    """80 filtered experimental triple helices, one .cif per PDB entry."""
    return os.path.join(data_root("experimental"), "experimental_cif")


def manifest_path(fmt: str = "csv") -> str:
    """Benchmark manifest: one row per PDB entry with per-chain sequences."""
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"fmt must be 'parquet' or 'csv', got {fmt!r}")
    root = data_root("experimental", patterns=(f"experimental/manifest.{fmt}",))
    return os.path.join(root, f"manifest.{fmt}")


def cdsm_dir(stage: str, trajectories: bool = False) -> str:
    """Deterministically built structures for one pipeline stage.

    The relaxed and annealed stages carry ~95 MB of MD trajectories in a
    trajectories/ subdirectory. Those are skipped unless asked for, so scoring
    fetches ~15 MB of structures rather than the whole stage.
    """
    if stage not in CDSM_STAGES:
        raise ValueError(f"unknown CDSM stage {stage!r}; expected one of {CDSM_STAGES}")
    ignore = () if trajectories else (f"cdsm/{stage}/trajectories/*",)
    return data_root(f"cdsm/{stage}", ignore=ignore)


def predictions_dir(model: str) -> str:
    """ML-predicted structures for one model, as <PDB>_<model>.cif."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    return data_root(f"predictions/{model}")


def scores_dir() -> str:
    """Scoring tables produced by 4_scoring/score.py."""
    return data_root("scores")


# ── write helper ─────────────────────────────────────────────────────────────
def local_out(*parts: str) -> str:
    """A directory inside the working tree for writing. Never the HF cache."""
    path = os.path.join(repo_root(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    print(f"repo_root: {repo_root()}\n")
    for prefix in sorted(LAYOUT):
        try:
            resolved = data_root(prefix)
            mark = "ok " if os.path.isdir(resolved) else "MISSING"
            print(f"  [{mark}] {prefix:38s} -> {os.path.relpath(resolved, repo_root())}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"  [ERR] {prefix:38s} -> {exc}")
