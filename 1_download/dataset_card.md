---
pretty_name: CDSM Collagen Structure Benchmark
license: other
license_name: mixed-see-licensing-section
tags:
  - biology
  - protein-structure
  - collagen
  - structure-prediction
  - benchmark
size_categories:
  - n<1K
configs:
  - config_name: manifest
    data_files:
      - split: train
        path: experimental/manifest.parquet
  - config_name: scores_summary
    data_files:
      - split: train
        path: scores/scores_summary.parquet
  - config_name: scores_per_residue
    data_files:
      - split: train
        path: scores/scores_per_residue.parquet
  - config_name: cdsm_stage_table
    data_files:
      - split: train
        path: scores/cdsm_stage_table.parquet
  - config_name: method_summary_table
    data_files:
      - split: train
        path: scores/method_summary_table.parquet
---

# CDSM Collagen Structure Benchmark — Data

Structures and scores for a benchmark comparing a deterministic collagen
triple-helix builder (CDSM) against four co-folding models — Boltz-2, Chai-1,
Protenix and AlphaFold3, the last in both with-MSA (`af3_msa`) and no-MSA
(`af3_nomsa`) conditions — on 80 experimentally resolved collagen triple
helices from the RCSB PDB.

Code: https://github.com/bm-howard/cdsm_benchmarking

## Layout

| Prefix | Contents | Size |
|---|---|---|
| `experimental/` | 80 filtered experimental triple helices (`.cif`) + `manifest.csv/.parquet` | 8.3 MB |
| `cdsm/<stage>/` | Deterministically built structures, one directory per pipeline stage | 16 MB |
| `cdsm/<stage>/trajectories/` | MD trajectories (`.dcd` + starting `.pdb`) for the relaxed and annealed stages | 95 MB |
| `predictions/<model>/` | `<PDB>_<model>.cif` for `boltz`, `chai`, `protenix`, `af3_msa`, `af3_nomsa` | 19 MB |
| `scores/` | Scoring tables, as both `.csv` and `.parquet` | 4.4 MB |

CDSM stages, in pipeline order: `coreonly`, `fullseq`, `fullseq_reregistered`,
`fullseq_reregistered_relaxed`, `fullseq_reregistered_annealed`.

The prefixes are deliberate: reproducing the scoring needs `experimental/`,
`cdsm/` and `predictions/` — about 40 MB — rather than the full 142 MB. The
trajectories are the bulk of the dataset and are almost never needed.

## Loading

The benchmark code resolves these paths for you:

```python
from data_locations import experimental_cif_dir, predictions_dir, cdsm_dir

exp = experimental_cif_dir()                      # downloads on first use
af3 = predictions_dir("af3_msa")
mdt = cdsm_dir("fullseq_reregistered_relaxed", trajectories=True)
```

Directly, without the repo:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="CollagenHelixLabs/cdsm_benchmarking_data",
    repo_type="dataset",
    allow_patterns=["predictions/**", "experimental/**"],   # skip the trajectories
)
```

The four tabular configs load as datasets:

```python
from datasets import load_dataset

scores = load_dataset("CollagenHelixLabs/cdsm_benchmarking_data", "scores_summary")
```

## Schemas

**`manifest`** — one row per PDB entry (80 rows):
`pdb_id`, `kind` (`homotrimer` | `heterotrimer`), `deposition_date`,
`n_distinct_chains`, `gly_start`, `frame_offset`, `has_hyp`, `len_a/b/c`,
`chain_a/b/c_sequence`. Sequences are one-letter codes with **`O` =
hydroxyproline (HYP)**.

**`scores_summary`** — one row per (structure, variant), 700 rows:
`pdb_id`, `variant`, `tm_score`, `global_rmsd_allatom`,
`global_rmsd_backbone`, `global_lddt_allatom`, `global_lddt_backbone`,
`coverage`.

**`scores_per_residue`** — long format, 55,684 rows:
`pdb_id`, `variant`, `chain`, `resnum`, `rmsd_allatom`, `rmsd_backbone`,
`lddt_allatom`, `lddt_backbone`.

**`cdsm_stage_table`** — mean/median per CDSM pipeline stage and metric,
written by `figures/make_figures.py`.

**`method_summary_table`** — mean/median per method and metric, formatted for
the manuscript. **A static snapshot, not a derived table**: no script in the
repository regenerates it, and it was built over a shared-target subset that
differs from the current one, so its values will not match `scores_summary`
exactly. Treat `scores_summary` as authoritative.

RMSD is computed in US-align's global-fit frame; lDDT is superposition-free.
TM-score is Cα, reference-normalised (`USalign -mm 1`).

## Coverage notes

- All five prediction variants cover all 80 targets. The CDSM stages cover 75: five
  entries (`1EI8`, `6M80`, `5K86`, `7LXQ`, `7LXP`) have internal Gly-X-Y
  register interruptions that divide by zero in the builder's propensity
  step, so they have no deterministic build at any stage.
- The annealed stage covers 19 structures, not 75 — it is a targeted
  comparison against the relaxed stage, not a full sweep.

## Licensing

**This dataset is mixed-licence. The prefixes are not interchangeable.**

| Prefix | Source | Terms |
|---|---|---|
| `experimental/` | RCSB PDB | CC0 — public domain |
| `cdsm/` | This work | See repository licence |
| `scores/` | This work | See repository licence |
| `predictions/af3_msa/`, `predictions/af3_nomsa/` | **AlphaFold Server (Google DeepMind)** | **Output Terms of Use — non-commercial only** |
| `predictions/boltz/`, `chai/`, `protenix/` | Generated locally with the respective open models | Each model's own licence |

`predictions/af3_msa/` and `predictions/af3_nomsa/` are AlphaFold Server Output.
Google DeepMind's Output Terms
of Use apply to it and to anything substantially derived from it, including a
non-commercial restriction. Anyone redistributing or building on that prefix is
bound by those terms; see https://alphafoldserver.com/terms. The scores in
`scores/` include `variant == "af3_msa"` and `"af3_nomsa"` rows derived from
that Output.

## Revision history

Structures are replaced in place rather than versioned, so changes that alter
scores are recorded here. Superseded files remain in the dataset's commit
history.

- **2026-08-14** — `predictions/boltz/` regenerated from a local Boltz run; all
  80 structures replaced. Median TM-score moved 0.913 → 0.914 and backbone RMSD
  1.138 → 1.085 Å. `scores/` was recomputed against the new structures.
- **2026-08-14** — AlphaFold3 prefixes renamed to name their MSA condition
  explicitly: `af3` → `af3_msa`, `AF3_no_MSA` → `af3_nomsa`. Scores were
  relabelled, not recomputed; the `af3_msa` values are unchanged from the
  earlier `af3` rows.

## Citation

<!-- TODO: replace on publication -->

```bibtex
@unpublished{cdsm_collagen_benchmark,
  title  = {Deterministic and co-folding model predictions of collagen triple helices},
  author = {Howard, Bruno},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

Please also cite the RCSB PDB, US-align, and whichever prediction models you use.
