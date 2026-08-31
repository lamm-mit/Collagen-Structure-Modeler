#!/usr/bin/env python3
"""
protenix_modal_app.py — Modal app that runs Protenix folding on a cloud CUDA GPU.

Protenix (ByteDance, https://github.com/bytedance/Protenix) is an open AF3-style
all-atom co-folding model with native CCD modified-residue support, so it encodes
hydroxyproline faithfully (unlike RFAA, whose modified-AA support is not yet
released). Same Modal pattern as the Chai app: driver `run_protenix.py` imports
`app`/`fold`, opens `app.run()`, and calls `fold.map(...)` per target.

Weights + CCD data auto-download on first use into a persistent Modal Volume
(mounted as $HOME so Protenix's ~/.protenix cache survives across runs).

One-time auth (user, needs a browser): python3.12 -m modal setup
"""

import modal

MINUTES = 60
CACHE_DIR = "/cache"                        # persistent cache (mounted as $HOME)
MODEL_NAME = "protenix_base_default_v1.0.0"

# L40S (48 GB) — proven-good tier for this model scale; drop to "A10G" to save cost.
GPU = "L40S"

app = modal.App("collagen-protenix")

weights_volume = modal.Volume.from_name("collagen-protenix-weights", create_if_missing=True)

# Protenix JIT-compiles a fused-LayerNorm CUDA kernel at import, so the image needs
# the full CUDA toolkit (nvcc + headers), not just torch's runtime — hence the CUDA
# *devel* base (12.6 matches the torch cu126 wheels protenix pulls) with CUDA_HOME set.
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu22.04", add_python="3.12")
    # Modal's add_python records `clang` as the compiler; some deps (scikit-learn-extra)
    # build C extensions from source, so install clang + a full toolchain.
    .apt_install("clang", "build-essential", "git")
    .pip_install("protenix")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)
# NB: HOME is set at RUNTIME inside fold() (not via image .env), otherwise the build
# populates /cache and Modal refuses to mount the Volume on a non-empty path.


CKPT_URL = ("https://protenix.tos-cn-beijing.volces.com/checkpoint/"
            f"{MODEL_NAME}.pt")
CKPT_PATH = f"{CACHE_DIR}/checkpoint/{MODEL_NAME}.pt"


@app.function(image=image, volumes={CACHE_DIR: weights_volume}, timeout=30 * MINUTES)
def setup_weights() -> str:
    """Robustly fetch the checkpoint into the Volume with a size check, then commit.

    Protenix's built-in downloader left a truncated .pt on interruption (torch then
    fails with 'failed finding central directory'). We download it ourselves with
    Content-Length verification so fold() always sees a complete file. The small
    data-cache files (CSV) are left to Protenix — those download fine.
    """
    import os
    from pathlib import Path
    import requests

    dst = Path(CKPT_PATH)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(CKPT_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers["Content-Length"])
        got = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
    if got != total:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete checkpoint: {got} != {total} bytes")
    weights_volume.commit()
    return f"checkpoint OK: {got/1e6:.0f} MB → {dst}"


@app.function(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: weights_volume},
    timeout=45 * MINUTES,     # cold run JIT-compiles cuequivariance kernels (then cached)
    max_containers=6,
)
def fold(name: str, job: dict, use_msa: bool = False) -> str:
    """Fold one Protenix job dict; return the top-ranked sample's CIF text.

    `job` is a single Protenix input record ({"name","sequences":[...]}). We wrap
    it in a list, run `protenix predict --use_msa false` (single-sequence, to match
    the Boltz/Chai empty-MSA setting), and return the best predicted CIF.
    """
    import glob
    import json
    import os
    import subprocess
    from pathlib import Path

    # Cache weights + CCD data on the mounted Volume (set here, not at build time,
    # so /cache stays empty for the mount). Protenix reads $HOME/.protenix.
    os.environ["HOME"] = CACHE_DIR

    work = Path("/tmp") / name
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = work / "input.json"
    in_path.write_text(json.dumps([job]))

    cmd = [
        "protenix", "pred",                      # installed CLI uses `pred` (not `predict`)
        "-i", str(in_path),
        "-o", str(out_dir),
        "-n", MODEL_NAME,
        "--seeds", "1",
        "--use_msa", "true" if use_msa else "false",
        "--sample", "1",                         # 1 sample/target — fair vs 1-sample Boltz/Chai
    ]
    print(f"[{name}] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)                    # stream stdout/stderr live into Modal logs
    weights_volume.commit()                       # persist freshly-downloaded weights + kernels

    cifs = sorted(glob.glob(str(out_dir / "**" / "*.cif"), recursive=True))
    if not cifs:
        raise RuntimeError(f"no CIF produced (protenix rc={proc.returncode}); see logs above")
    # Prefer a rank-0 / sample-0 file if the naming exposes ranking; else first.
    best = next((c for c in cifs if "sample_0" in c or "rank_0" in c or "_0." in c), cifs[0])
    print(f"[{name}] produced {len(cifs)} cif(s); chose {os.path.basename(best)}")
    return Path(best).read_text()
