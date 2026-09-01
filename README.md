# CDSM Collagen Structure Benchmarking

Benchmark comparing a deterministic collagen triple-helix builder (CDSM)
against four co-folding models — Boltz-2, Chai-1, Protenix and AlphaFold3,
the last in both with-MSA and no-MSA conditions — on 80 experimentally resolved collagen triple helices from the RCSB PDB.

Data lives separately, on HuggingFace:
[`CollagenHelixLabs/cdsm_benchmarking_data`](https://huggingface.co/datasets/CollagenHelixLabs/cdsm_benchmarking_data).
You do not need to download it by hand — the code fetches what it needs.

## Layout

```
0_data/                    experimental structures + manifest   (not in git)
1_download/                RCSB query, filtering, HF upload + dataset card
2_deterministic_build/     CDSM pipeline, step1 - step5
3_ml_prediction/           per-model runners + Modal apps; shared/ also holds
                           the seed-sensitivity sweep machinery and analysis
4_scoring/                 US-align / lDDT / TM-score scoring
figures/                   figure generation + figures_png/, figures_pdf/, other_figs/
notes/                     PAPER_METHODS.md - consolidated methods write-up
data_locations.py          where the data lives - see below
```

Directories holding data (`0_data/`, `*/outputs/`, `4_scoring/results/`) are
gitignored; they are populated from HuggingFace on demand.

## Setup

```bash
git clone https://github.com/lamm-mit/Collagen-Structure-Modeler.git
cd Collagen-Structure-Modeler
pip install -r requirements.txt
bash scripts/get_usalign.sh          # builds the US-align binary into tools/
```

The dataset is currently private, so you will also need read access and a
login:

```bash
hf auth login
```

## Running

```bash
python 4_scoring/score.py                  # score everything -> 4_scoring/results/
python 4_scoring/score.py --pdb-id 8K4X    # one structure
python figures/make_figures.py             # regenerate all figures
python 2_deterministic_build/run_pipeline.py
```

Scoring downloads ~43 MB on first run and caches it in `~/.cache/huggingface`.
Subsequent runs hit the cache.

`make_figures.py` writes every paper figure as both PNG and PDF into
`figures/figures_png/` and `figures/figures_pdf/`, under the manuscript's own
numbering; exploratory figures go to `figures/other_figs/` as PNG only. To
promote a figure, add it to `PAPER_FIGURES` in the script. The PDFs are
gitignored, since they regenerate on every run.

## Where the data comes from

Every read goes through [`data_locations.py`](data_locations.py), which
resolves a dataset prefix to a real directory in one of three ways, in order:

1. `COLLAGEN_DATA_ROOT`, if set — a local copy in HuggingFace layout.
2. The working tree, if the corresponding directory exists. This is what
   happens on a machine that produced the data.
3. HuggingFace, downloaded lazily per-prefix.

All three return a plain directory path, so the scripts themselves contain no
HuggingFace-specific code.

```python
from data_locations import experimental_cif_dir, predictions_dir, cdsm_dir

exp = experimental_cif_dir()
af3 = predictions_dir("af3_msa")
mdt = cdsm_dir("fullseq_reregistered_relaxed", trajectories=True)
```

Useful environment variables:

| Variable | Effect |
|---|---|
| `COLLAGEN_DATA_ROOT` | Read from this local directory instead of the Hub |
| `COLLAGEN_FORCE_HUB=1` | Ignore the working tree; read only from the Hub |

`COLLAGEN_FORCE_HUB=1` is the way to check that what was uploaded reproduces
what you have locally:

```bash
COLLAGEN_FORCE_HUB=1 python 4_scoring/score.py
```

Writes never go to HuggingFace. Scripts write into the working tree, and
uploading is a separate deliberate step.

## Uploading data

```bash
python 1_download/upload_to_huggingface.py --dry-run --all   # plan only
python 1_download/upload_to_huggingface.py --all --trajectories
python 1_download/upload_to_huggingface.py --section scores
python 1_download/upload_to_huggingface.py --card            # dataset card only
```

Uploads are additive; `--replace` clears a prefix first. The working-tree ->
dataset-prefix mapping is the `LAYOUT` table in `data_locations.py`, used by
both the readers and the uploader, so a new section is declared once.

The dataset card lives at `1_download/dataset_card.md` and is published as the
dataset repo's `README.md` by `--card` (included in `--all`). Edit it there
rather than in the HuggingFace web UI, so it stays versioned with the code.

Trajectories (~95 MB) are uploaded only with `--trajectories`, and are never
downloaded unless explicitly requested.

## Notes and caveats

- Five entries (`1EI8`, `6M80`, `5K86`, `7LXQ`, `7LXP`) have internal Gly-X-Y
  register interruptions that divide by zero in the builder's propensity step,
  so the CDSM stages cover 75 of 80 structures. All five prediction variants
  cover all 80.
- The annealed stage covers 19 structures — a targeted comparison against the
  relaxed stage, not a full sweep.
- Sequences use one-letter codes with `O` = hydroxyproline (HYP).
- **`predictions/af3_msa/` and `predictions/af3_nomsa/` are AlphaFold Server
  Output and carry Google
  DeepMind's non-commercial Output Terms of Use.** See the dataset card.

Methods and filtering decisions are written up in [`notes/`](notes/).

## Licensing

The **code** in this repository is licensed under Apache-2.0 (see `LICENSE`).

The **data** on HuggingFace is mixed-licence and Apache-2.0 does not extend to
it. In particular, `predictions/af3_msa/` and `predictions/af3_nomsa/` are
AlphaFold Server Output and carry Google DeepMind's Output Terms of Use,
including a **non-commercial restriction**; `experimental/` is CC0 from the RCSB
PDB. See the [dataset card](1_download/dataset_card.md) for the per-prefix
breakdown before redistributing or building on any of it.
