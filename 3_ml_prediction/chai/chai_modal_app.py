#!/usr/bin/env python3
"""
chai_modal_app.py — Modal app that runs Chai-1 folding on a cloud CUDA GPU.

This file defines the remote pieces only (container image + GPU function). The
benchmark driver `run_chai.py` imports `app` and `fold` from here, opens an
ephemeral `app.run()`, and calls `fold.remote(...)` / `fold.map(...)` per target.

Chai-1 runs on CUDA (not the M1's Metal), so all inference happens in Modal
containers. Model + ESM weights auto-download on first use into a persistent
Modal Volume (CHAI_DOWNLOADS_DIR), so only the first structure pays for it.

Container recipe (torch cu128 + chai_lab 0.5.0) matches Chai/Modal's own tested
example: https://modal.com/docs/examples/chai1

One-time auth (done by the user, needs a browser):
    python3.12 -m modal setup
"""

import modal

MINUTES = 60
CACHE_DIR = "/cache"                       # persistent weights cache (Modal Volume)

# GPU choice: collagen inputs are tiny (~90 tokens). L40S (48 GB) is a proven-working
# config for chai_lab on Modal and only marginally pricier across the batch; drop to
# "A10G" (24 GB, cheaper — also enough) to save a little, or "H100" for max speed.
GPU = "L40S"

app = modal.App("collagen-chai1")

# Persist model + ESM weights across runs so only the first fold downloads them.
weights_volume = modal.Volume.from_name("collagen-chai1-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chai_lab==0.6.1", "huggingface-hub")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    # Chai downloads BOTH its model files and the ESM2-3B embedding weights into
    # CHAI_DOWNLOADS_DIR, so pointing it at the mounted Volume caches everything;
    # only the first fold pays the ~10 GB download.
    .env({"CHAI_DOWNLOADS_DIR": f"{CACHE_DIR}/chai"})
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: weights_volume},
    timeout=15 * MINUTES,
    max_containers=6,          # cap parallel GPUs during a --all batch to bound cost
)
def fold(name: str, fasta: str, use_msa: bool = False) -> str:
    """Fold one FASTA string with Chai-1; return the best model's CIF text.

    `use_msa=False` (default) is single-sequence mode (ESM embeddings, no MSA) —
    best for collagen's repetitive Gly-X-Y, matching the Boltz empty-MSA result.
    We generate 1 diffusion sample (to match the benchmark's 1-sample Boltz run);
    the aggregate_score selection below stays general if that count is raised.
    """
    import glob
    import re
    import shutil
    from pathlib import Path

    import numpy as np
    import torch
    from chai_lab.chai1 import run_inference

    workdir = Path("/tmp") / name
    out_dir = workdir / "out"
    shutil.rmtree(workdir, ignore_errors=True)      # run_inference wants a clean out dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = workdir / "input.fasta"
    fasta_path.write_text(fasta)

    run_inference(
        fasta_file=fasta_path,
        output_dir=out_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_diffn_samples=1,        # match our Boltz run (1 sample/target) for a fair benchmark
        seed=42,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
        use_msa_server=use_msa,
    )
    # Persist any newly downloaded weights for subsequent containers/runs.
    weights_volume.commit()

    # Pick the top-ranked of the 5 diffusion samples by Chai's aggregate_score.
    best_idx, best_score = None, -float("inf")
    for npz in glob.glob(str(out_dir / "scores.model_idx_*.npz")):
        idx = int(re.search(r"model_idx_(\d+)", npz).group(1))
        try:
            score = float(np.asarray(np.load(npz)["aggregate_score"]).mean())
        except Exception:                            # noqa: BLE001 — fall back on odd npz
            score = -float("inf")
        if score > best_score:
            best_idx, best_score = idx, score
    if best_idx is None:                             # no scores parsed → use sample 0
        best_idx = 0

    cif_path = out_dir / f"pred.model_idx_{best_idx}.cif"
    return cif_path.read_text()
