# Methods — Collagen Triple-Helix Structure Prediction Benchmark

Consolidated methods reference for the CDSM benchmark: dataset construction,
deterministic (physics-based) structure generation, deep-learning prediction,
scoring methodology, and results. Parameter values, tool versions, decision
rationale and known limitations are included.

This document supersedes and combines the previous `NOTES.md`,
`data_filtering.md`, `METHODS_ml_benchmark.md`, `CDSM_scoring_handoff.md` and
`RESULTS_ml_benchmark.md`.

**All results tables were recomputed from `4_scoring/results/scores_summary.csv`
(700 rows) on 2026-08-15.** Numbers quoted in the superseded documents came from
three earlier scoring generations and should not be reused — see §9.1.

---

## 1. Overview

We benchmark methods for predicting the 3D structure of collagen triple helices
against experimentally resolved structures from the RCSB PDB. Two method
families are compared:

- **Deterministic / physics-based (CDSM)** — an idealized THeBuScr backbone
  builder extended with terminal extension, register correction, side-chain
  construction (AmberTools `tleap`, ff14SB) and OpenMM energy minimization.
- **Deep-learning** — Boltz-2, Chai-1, Protenix-v1 and AlphaFold3, the last in
  both with-MSA and no-MSA conditions.

**Collagen-specific considerations** throughout: chains are homotrimers (one
distinct sequence, assembled ×3) or heterotrimers (three distinct sequences),
always three chains; **hydroxyproline (HYP, one-letter `O`)** is a modified
residue that must be encoded and parameterized explicitly; and the inter-chain
one-residue stagger/register is a structural degree of freedom.

**Software:** gemmi 0.7.5, NumPy 1.26.4, ParmEd 4.3.1, OpenMM 8.4, AmberTools
(`tleap`, ff14SB), US-align (v20260527), `boltz-api` CLI 0.37.1 (`boltz-2.1`),
`chai_lab` 0.6.1, `protenix` 2.0.0 (checkpoint `protenix_base_default_v1.0.0`).

---

## 2. Dataset construction

### 2.1 Source and search

- **Database:** RCSB Protein Data Bank
- **API:** RCSB Search API v2 (`https://search.rcsb.org/rcsbsearch/v2/query`)
- **Format:** mmCIF, one file per structure
- **Script:** `1_download/download_collagen.py`

Three AND-joined terminal queries:

| Filter | Value | Rationale |
|---|---|---|
| Full-text keyword | `"collagen"` OR `"triple helix"` | Broad initial net; matches title, entity description, keywords |
| `deposited_polymer_entity_instance_count` | ∈ {3, 6, 9, 12} | Multiples of 3 capture structures with >1 triple helix in the asymmetric unit |
| `selected_polymer_entity_types` | `"Protein (only)"` | Excludes nucleic-acid complexes and ligand-only entries |

**Initial candidate pool:** 758 entries (varies slightly as the PDB index updates).

**Why multiples of 3, not exactly 3?** An early query required exactly 3 chain
instances, which excluded structures such as `8K4X` (human collagen XVII),
deposited as 6 chains — two copies of a triple helix in the ASU, a common
crystallographic practice. The upper bound of 12 is pragmatic; no known collagen
crystal structure is expected to exceed 4 triple helices per ASU.

### 2.2 Structural filters

Structures are parsed with `gemmi`; protein chains are identified by the presence
of at least one Cα. Filters are applied in the order below; failure at any step
rejects the whole structure and logs the reason to `1_download/rejected.log`.

**Ordering rationale.** The glycine-fraction check is placed early (immediately
after the residue allowlist) so that non-collagen proteins — the largest
rejection category — are attributed to that filter rather than to chain-length or
gap checks, which would misrepresent the data. The ordering reflects biological
significance, not computational cost.

**2.2.1 ATOM chain count is a positive multiple of 3.** Confirmed from ATOM
records after download, not from SEQRES or API metadata alone. A secondary
confirmation of the API-level filter. *(2026-07-14 run: 0 rejected — the API
filter held.)*

**2.2.2 Allowed residues only.** Every observed residue must lie in the
**21-residue allowed set**: the 20 standard amino acids plus
**(4R)-hydroxyproline (HYP)**. Any other residue rejects the structure.

*Rationale:* the deterministic pipeline can only generate these 21 residue types.

Rejected residue types encountered:

| Code | Name | Why excluded |
|---|---|---|
| DPR | D-proline | Non-natural stereoisomer |
| HZP | (4S)-4-hydroxy-L-proline | *allo*-HYP isomer; structurally distinct from (4R)-HYP |
| HY3 | 3-hydroxyproline | Different hydroxylation position |
| NLY | N-(4-aminobutyl)glycine | Synthetic Gly analogue |
| SAR | Sarcosine (N-methylglycine) | N-methylated Gly |
| 4BF | 4-bromo-L-phenylalanine | Halogenated synthetic residue |
| MSE | Selenomethionine | Phasing derivative of Met |
| LYZ | 5-hydroxylysine | Collagen-relevant but outside allowed set |
| 2L6 | Spiro-fused proline analogue | Synthetic |

HZP and HY3 are chemically close to HYP and are genuine collagen
post-translational modifications; their exclusion is a real constraint on
benchmark coverage, particularly for basement-membrane collagens.
*(2026-07-14 run: 136 rejected.)*

**2.2.3 Glycine fraction ≥ 0.25 per chain.** The canonical Gly-X-Y repeat
predicts ~33.3%; the threshold is lower to accommodate termini that do not
complete a triplet.

*Rationale:* the strongest single collagen-specificity filter. Non-collagen
3-chain proteins matching the keyword search (trimeric antibodies, coiled-coils,
viral coat proteins) typically carry ~7–11% glycine.

*Threshold history:* 40% (too strict, rejected genuine collagens) → 27% → 25%.
*(2026-07-14 run: 530 rejected — the single largest filter.)*

**2.2.4 Internal residue-numbering gap ≤ 5.** Consecutive residue numbers within
a chain must not jump by more than 5. Larger jumps indicate a true structural
discontinuity (internal missing density) rather than terminal truncation or a
numbering convention. This is the pipeline's only completeness check.
*(2026-07-14 run: 1 rejected.)*

**2.2.5 Chain length 10–200 residues** (observed ATOM residues). The lower bound
excludes tiny synthetic model peptides of limited benchmarking value (e.g.
`1A3I`, chains of 6–9 residues); the upper bound is a pipeline capability limit.

*Threshold history:* originally 20–100, widened to 10–200. After moving the Gly
check earlier, zero structures fail the upper bound — the previous 149 "too long"
rejections were almost entirely non-collagen proteins now correctly captured by
the Gly filter. *(2026-07-14 run: 7 rejected, all on the lower bound.)*

**2.2.6 ATOM coverage ≥ 95% of SEQRES — REMOVED 2026-08-28.** Deleted from
`download_collagen.py`; it should not be re-enabled.

*Why it never fired.* `entity_seqres` was keyed by subchain-name strings, but the
lookup passed `scs[0]`, a `gemmi.ResidueSpan`. The lookup always missed,
`seqres_len` fell back to `atom_len`, and coverage was therefore always exactly
100%. The "0 structures rejected" previously recorded was an artefact of that
bug, not evidence of completeness.

*Why removed rather than fixed.* With the lookup corrected, a 95% threshold
removes **51 of the 80** retained structures. The cause is chain length, not data
quality: these peptides are ~25–30 residues, so one unmodelled terminal residue
is ~96% and two are ~93%. A 95% cutoff is designed for ~300-residue globular
proteins; on a 27-mer it means "zero missing termini" (`6HG7`, 36/38 = 94.7%,
fails on rounding alone). It also measures the wrong thing — of all 80 structures
only **7** have any internally missing residue (`7JX5`, `5K86`, `9PUF`: 1
residue; `6M80`, `7LXP`, `7LXQ`, `7VEG`: 2). Every other deficit is terminal
disorder, harmless here because sequences are ATOM-derived (§2.4), so a missing
terminus simply yields a slightly shorter benchmark target. Internal breaks are
already handled by §2.2.4.

**2.2.7 Exact-sequence duplicate removal.** Structures whose retained triplet has
an **identical multiset of chain sequences** are collapsed to one representative.
The comparison is over all three chains (the sequence *signature*), not chain A
alone, so structures sharing one chain but differing in another are correctly
kept as distinct.

The representative is chosen by a deterministic hierarchy:
1. **Lowest (best) resolution** — unknown resolution always loses.
2. **Fewest chains in the ASU** — prefer a single triple helix.
3. **Lexicographically smallest PDB ID** — final tie-break.

*(2026-07-14 run: 4 removed across 3 groups — `9IBU` kept (1.25 Å) over `9I99`,
`9I9A`; `6VZX` kept (1.37 Å) over `3T4F` (1.68 Å); `5Y46` kept (1.03 Å, 3 chains)
over `5Y45` (1.03 Å, 6 chains, decided on chain count). `5YAN`, `8GZO`, `8H0E`
share chain A but differ in B/C, so they are not exact duplicates and are all
retained.)*

### 2.3 Candidate funnel (run 2026-07-14)

| Step | Filter | Removed | Remaining |
|---|---|---|---|
| API search | Keyword + chain count ∈ {3,6,9,12} + protein-only | — | **758** |
| 2.2.1 | ATOM chain count not a positive multiple of 3 | 0 | 758 |
| 2.2.2 | Contains unhandled residue(s) | 136 | 622 |
| 2.2.3 | Gly fraction < 25% | 530 | 92 |
| 2.2.4 | Internal residue gap > 5 | 1 | 91 |
| 2.2.5 | Chain too short (< 10 residues) | 7 | 84 |
| 2.2.5 | Chain too long (> 200 residues) | 0 | 84 |
| 2.2.7 | Exact-sequence duplicate | 4 | **80** |

**Final dataset: 80 structures** (30 homotrimers, 50 heterotrimers), all with
experimental coverage 1.0.

### 2.4 Multi-copy ASU handling and sequence representation

For structures with 6, 9 or 12 chains, only the **first triplet** (the first three
protein chains in ATOM-record order, typically A/B/C) is retained; the rest are
discarded before saving. Multiple ASU copies arise from crystallographic packing
rather than biological relevance. The saved file is always `{PDB_ID}.cif`
containing only the 3 retained chains.

Sequences are extracted from **ATOM records**, not SEQRES. SEQRES represents the
author-deposited intended sequence, which may include terminal residues absent
from the crystal and does not distinguish modified residues usefully; ATOM
records reflect what was actually observed.

Hydroxyproline is retained as the distinct one-letter code **`O`**, not collapsed
to `P`. `O` is the recognised single-letter extension for (4R)-hydroxyproline and
preserves the chemical distinction the deterministic pipeline requires. *(An
initial version collapsed HYP→`P` and HYL→`K`, losing chemical specificity; this
was corrected.)*

### 2.5 Outputs

`0_data/experimental_cif/<PDB>.cif` plus `0_data/manifest.csv` with columns:
`pdb_id, kind, deposition_date, n_distinct_chains, gly_start, frame_offset,
has_hyp, len_a/b/c, chain_a/b/c_sequence`.

Mirrored to the HuggingFace dataset `CollagenHelixLabs/cdsm_benchmarking_data`
under `experimental/`.

### 2.6 Dataset assumptions

- **Chain ordering is meaningful.** The first three protein chains (A/B/C) are
  assumed to form a biologically coherent triple helix — standard PDB convention,
  but not guaranteed for all depositors.
- **HYP always denotes (4R)-hydroxyproline**, the form predominant in natural
  collagen. Other stereoisomers use distinct codes (HZP, HY3) and are excluded.
- **Gly-X-Y register is not explicitly verified.** Glycine fraction is a proxy; a
  chain with ≥25% Gly but a non-Gly-X-Y arrangement would pass. Unlikely given the
  keyword filter and chain-count constraint acting together.
- **No resolution or R-factor filter.** Structures of any resolution and
  experimental method are accepted if they pass the structural filters. This
  maximises dataset size at the cost of uniform quality.
- **The ASU is treated as the biological assembly.** The deposited ASU is used
  directly rather than the curated `assembly_1.cif`. Equivalent for most collagen
  peptide structures, but not guaranteed for all entries.

### 2.7 Open questions on the dataset

1. **Should a resolution cutoff be applied?** None is currently used; ~2.5 Å is
   standard in structural benchmarking and would exclude low-quality references.
2. **Are all 80 structures truly independent?** Exact-sequence duplicates are
   removed, but *near*-identical sequences solved under different conditions
   remain. Clustering by sequence identity (e.g. 90%) may be appropriate,
   especially for the engineered host-guest peptide series that dominate the
   redundancy. The shared collagen scaffold inflates full-sequence identity, so
   identity computed over the variable (non-`GPO`/`GPP`/`GOO`) region is the more
   informative diversity measure.
3. **Is first-triplet-only appropriate for all multi-copy ASUs?** The choice of
   A/B/C is arbitrary when copies differ; selecting the most complete triplet is
   an alternative. For `8K4X` the two copies differ in terminal coverage (25–26 vs
   26–27 residues per chain).
4. **Heterotrimer chain assignment.** Chain identities are preserved as deposited
   but not verified against biological convention (e.g. α1/α1/α2 ordering). Some
   predictors may be sensitive to chain input order.
5. **Rejection double-counting.** Because filters are sequential, the 136
   unhandled-residue rejections may include structures that would also fail the
   Gly check. The count of genuine collagens lost *solely* to the residue
   constraint is unknown without running both checks independently.

---

## 3. Deterministic structure generation (CDSM)

A five-stage pipeline in `2_deterministic_build/`. `run_pipeline.py --all` builds
every variant for every manifest entry; `--pdb-id X` / `--list f` for subsets.
Each stage was validated independently, and cumulative output variants were
produced per structure to isolate each capability's effect.

| Step | File | What it does |
|---|---|---|
| 1 | `step1_backbone_builder.py` | Byte-exact THeBuScr port; `reframe()` + core/extend completion modes |
| 2 | `step2_terminal_extension.py` | `extend_termini()` — Kabsch screw from the immediately-adjacent terminal pair |
| 3 | `step3_register_fix.py` | `reregister()` — whole-triplet shifts minimising axial terminal overhang |
| 4 | `step4_sidechain_builder.py` | tleap ff14SB side chains; ParmEd chain-ID restoration; MD-ready heavy-atom CIF |
| 5 | `step5_relax.py` | OpenMM OBC2 minimization under backbone restraints |

### 3.1 Backbone (THeBuScr port)

Backbones are built with a pure-Python port of THeBuScr (Rainey & Goh 2004),
verified byte-for-byte against the original on 533 sequences. THeBuScr constructs
an idealized cylindrical-polar triple helix from propensity-averaged geometry
parameters (two symmetry classes, 7/2 "AR" and 10/3 "IR"), placing backbone atoms
(N, CA, C, O) from each residue's position in the Gly-X-Y triplet and an assigned
helix type. Backbone geometry is independent of side-chain identity.

THeBuScr requires each chain in Gly-X-Y register, a length that is a multiple of
three, and near-equal chain lengths — conditions deposited sequences rarely meet.
A preprocessing layer therefore:

- **reframes** each chain to the reading frame with the most glycines at triplet
  starts (dropping ≤2 leading residues and any trailing partial triplet);
- offers two completion modes: **core** (truncate all chains to the common
  equal-length core; pure THeBuScr) and **extend** (propagate the terminal helix
  type over the whole-triplet overhang of longer chains, so no residue is lost).

**Limitation.** 5 structures (`1EI8`, `6M80`, `5K86`, `7LXQ`, `7LXP`) fail with a
divide-by-zero in THeBuScr's propensity averaging. These have **internal
interruptions of the Gly-X-Y periodicity** (indels, not phase shifts), so a
propensity-averaging window contains no glycine-led triplet. A single global
reframe cannot fix a mid-chain register break, so these are not built.

### 3.2 Terminal overhang extension

`reframe` trims terminal residues to reach whole triplets (e.g. 1BKV chain A
loses a leading HYP and trailing GLY). These are restored by geometric
extrapolation: the rigid-body screw transform between two terminal residues'
backbone atoms is derived by Kabsch superposition and applied outward (and its
inverse at the N-terminus). This recovers the deposited residue count.

**Why the LOCAL adjacent pair, not a position-matched one.** Replacing the
adjacent-pair transform with a **position-matched** transform (sampling a
same-junction-type internal pair, so an appended Gly is placed by a real Y→Gly
step) was implemented, tested and reverted:

- Junction Cα–Cα rises differ by only ~0.1 Å (G→X 3.82, X→Y 3.78, Y→G 3.68), so
  position type barely matters.
- The transform is a **Kabsch fit on 4 backbone atoms, not a pure helix screw.**
  It degrades the farther it is extrapolated from where it was sampled; error
  grows monotonically (3.8, 5.8, 11.9, 19.7, 26.6, 31.1, 32.8 Å at increasing
  distance), tested on the *uniform* chain 1K6F, so this is fundamental rather
  than an interruption effect. Root cause: consecutive residues (Y vs G) have
  different backbone shapes, so the least-squares transform between them is
  orientation-dependent.
- The **inverse** direction (N-terminal prepend) is additionally worse than
  forward at equal distance (4.54 vs 3.68 Å), though correctly computed
  (round-trips to 0.09 Å).

**Conclusion: locality beats position-matching.** The adjacent-pair method
(1-residue extrapolation, no inversion, at both termini) is retained; energy
minimization erases the residual anyway.

### 3.3 Register correction

The builder applies a fixed A→B→C one-residue stagger. When chains differ in
length, the correct leading/trailing strand can be a whole triplet out of
register, leaving one strand splayed at a terminus. This is corrected by trying
integer **whole-triplet** shifts of each chain (which preserve Gly-X-Y: G→G, X→X,
Y→Y) and keeping the threading that **minimizes the axial spread of the chain
termini**. The one-residue stagger transform is derived from a reference
poly-(Gly-Pro-Hyp) homotrimer and is register-independent. A shift is applied only
if it reduces overhang by ≥ `MIN_IMPROVEMENT_A` = 2.0 Å.

On the dataset **25 of 75 structures were re-threaded**. The correction is net
positive but non-monotonic: **9 improved substantially** (1WZB +0.38, 2DRX +0.35,
4Z1R +0.35, 3A0M +0.35, 3A1H +0.33, 3A08 +0.33, 8K4Y +0.33, 8K4X +0.31 all-atom
lDDT), ~61 unchanged, and **5 regressed** (8TW0 −0.29, 8HHK −0.26 the notable
cases, where chains differ by multiple triplets and minimal-overhang is not the
correct register). Both corrected and uncorrected variants are retained.

### 3.4 Side-chain construction

Side chains are added with AmberTools `tleap` (ff14SB): `O` residues are renamed
to HYP, `tleap` builds all side-chain heavy atoms and hydrogens from residue
templates (**parameterizing HYP**), ParmEd restores chain IDs, and a heavy-atom
mmCIF is written (hydrogens stripped; HYP `_chem_comp_bond` topology injected for
downstream MD-readiness). Side chains are placed in library/default rotamers.
`add_sidechains(bb, out_cif, amber_out=prefix)` exports `{prefix}.prmtop` and
`.rst7` for step 5.

### 3.5 Energy minimization

The tleap-generated Amber topology and coordinates — which already carry HYP
parameters — are loaded into OpenMM 8.4 and minimized with:

- **OBC2 implicit solvent.** `GBn2` was rejected: it requires an mbondi3 radius
  set the default prmtop lacks and flags a HYP atom; OBC2 works with tleap's
  default radii. `nonbondedMethod = NoCutoff`, `constraints = HBonds`.
- A **harmonic positional restraint on backbone heavy atoms** (N, CA, C, O),
  k = 1000 kJ·mol⁻¹·nm⁻² (~2.4 kcal·mol⁻¹·Å⁻²), holding the fold while side
  chains relax.
- A **single continuous** `minimizeEnergy` to the default force tolerance (capped
  at 2000 iterations). Chunked minimization cold-restarts L-BFGS and stalls
  prematurely; a single continuous run converges deeply (~375 iterations for some
  structures).

Minimization moves atoms only ~0.26 Å RMSD on average (localized clash relief;
worst inter-chain contact typically 1.4 Å → 2.8 Å). Trajectories are captured with
an OpenMM `MinimizationReporter` (one frame per 5 iterations) as DCD plus a
chain-annotated reference PDB, written from the *minimised* coordinates via ParmEd
so chains and TER records are correct.

**Simulated annealing was tested and rejected.** To test whether the side-chain
gap is a *sampling* limitation, minimization was replaced by backbone-restrained
simulated annealing (OpenMM Langevin MD, heat to 600 K → cool through 450/300/150
K, 3 seeds, lowest-energy kept). On 19 structures (1 of 20 failed with an
integrator NaN on a clashy input), annealing gave **identical mean lDDT** to
minimization (all-atom 0.901/0.901, backbone 0.950/0.950; per-structure:
annealing better 3, minimization better 1, tie 15). Annealing samples alternative
rotamers, but every seed relaxes back to the same ff14SB-preferred minimum, which
does not match the crystal better.

**Interpretation: the deterministic side-chain gap is a force-field-accuracy
limitation, not a sampling limitation.** Additional conformational sampling cannot
cross the ceiling set by the energy function. Annealing was also ~2.5× slower and
less robust, so minimization was retained.

### 3.6 Output variants and build statistics

Output folders under `2_deterministic_build/outputs/`: `gen_struct_coreonly`,
`gen_struct_extendedchains`, `gen_struct_fullseq`,
`gen_struct_fullseq_reregistered`, `gen_struct_fullseq_reregistered_relaxed`
(+ `gen_struct_fullseq_reregistered_annealed`), each with `trajectories/*.dcd,*.pdb`
where applicable.

**Build statistics:** 75/80 built; 5 skipped (§3.1). 25 re-threaded by step 3.

Structure categories (of 80; overlapping, not a partition):
1. THeBuScr builds natively: **75** — of which only **7 fully native** (no trim,
   equal reframed lengths): `3B0S 1YM8 4DMT 3P46 5Y46 3POD 7VEG`.
2. Re-threaded after reframing: **25**.
3. `_extend_heltype` (whole-triplet overhang): **19**.
4. Kabsch extrapolation (partial-triplet trim): **68**.

---

## 4. Deep-learning prediction

### 4.1 Common input-encoding principles

Every model receives the same biological specification in its native format:

- **Stoichiometry.** Always 3 chains. Homotrimer = one protein entity with copy
  count 3; heterotrimer = three separate entities, one per chain.
- **Hydroxyproline.** The parent sequence carries `P` (proline) at each HYP site,
  with the modification declared separately via the PDB Chemical Component
  Dictionary code **HYP**. *The position-indexing convention differs by model and
  was the single most error-prone detail* — see §4.3.
- **MSA.** Single-sequence wherever the model allows it (§4.4).
- **Samples.** One prediction per target wherever controllable (§4.5).

All self-run models were executed on cloud GPUs via **Modal** (serverless,
pay-per-GPU-second); AlphaFold3 used the AlphaFold Server web interface.

### 4.2 Per-model configuration

**Boltz-2.** `boltz-api` CLI 0.37.1, model `boltz-2.1`, OAuth. HYP: sequence
carries `P`; modification `{"type":"ccd","residue_index":<0-indexed>,
"value":"HYP"}`. Homotrimer = one entity with `chain_ids:["A","B","C"]`;
heterotrimer = three entities. MSA `{"type":"empty"}` — only `empty` and `custom`
are valid `msa.type` values; automatic generation is the default (omit the field).
Schemas were validated with the free `estimate-cost` endpoint (no GPU spend);
idempotency keys (= PDB ID) prevented duplicate billing on re-runs.

**Chai-1.** `chai_lab==0.6.1` (Chai-2 weights are gated and were not used),
`torch==2.7.1` (CUDA 12.8 wheels), GPU L40S. Input FASTA, one record per chain,
headers `>protein|name=A|B|C`. **HYP written inline as the CCD code in
parentheses**, e.g. `...P(HYP)G...`. Inference (`chai_lab.chai1.run_inference`):
`num_trunk_recycles=3`, `num_diffn_timesteps=200`, `num_diffn_samples=1`,
`use_esm_embeddings=True`, `use_msa_server=False`, `seed=42`.

**Protenix-v1.** `protenix==2.0.0`, checkpoint `protenix_base_default_v1.0.0`
(368 M parameters — note the package version is the software, not the model),
GPU L40S. Built on a CUDA 12.6 *devel* image because Protenix JIT-compiles a
fused-LayerNorm CUDA kernel at import. Input JSON `proteinChain` entities;
homotrimer = one chain with `"count":3`. HYP via
`modifications:[{"ptmType":"CCD_HYP","ptmPosition":<1-indexed>}]`. CLI:
`protenix pred -i <in.json> -o <out> -n protenix_base_default_v1.0.0 --seeds 1
--use_msa false --sample 1`. Default recipe retained: **10 Pairformer recycles**,
200 diffusion steps, bf16 — the higher recycle count vs Chai's 3 is Protenix's
recommended default and the main reason its runtime is longer.

**AlphaFold3 (AlphaFold Server).** Public web service; jobs uploaded as JSON
(`dialect:"alphafoldserver"`, `version:1`). HYP via `ptmType:"CCD_HYP"`,
`ptmPosition` 1-indexed. **`useStructureTemplate:false` on every chain** (§4.6).
`modelSeeds:[]` (server assigns a single random seed). Server limit of 30 jobs/day
→ three stratified batches (each a representative homo/hetero mix) over three
days. AF3 returns **5 predictions per job** ordered by `ranking_score`; we take
`model_0` (§4.5).

Two AF3 conditions are reported:
- **`af3_msa`** — AlphaFold Server with its own MSA pipeline (the server always
  runs it; single-sequence cannot be selected).
- **`af3_nomsa`** — the no-MSA condition, matching the other models.

### 4.3 Input validation procedure

Because HYP encoding is the dominant failure mode, each model was validated
before any batch run:

1. Build the input for one HYP-containing target.
2. Cheaply validate the schema (a single smoke prediction).
3. **Smoke-test one structure and confirm HYP lands at the correct residue
   positions and that 3 chains A/B/C are present** — parse the output CIF,
   reconstruct each chain's one-letter sequence (mapping HYP→`O`), and check it
   matches the target sequence and HYP count.
4. Only then run the full 80.

This caught a 0- vs 1-indexed discrepancy between models: Boltz uses 0-indexed
`residue_index`; Chai uses inline `(HYP)`; Protenix and AF3 use 1-indexed
`ptmPosition`.

### 4.4 Single-sequence (no MSA)

Collagen's repetitive Gly-X-Y motif yields weak, uninformative MSAs. An
empty-vs-automatic MSA pilot with Boltz-2 on 5 structures (3 homo-, 2
heterotrimer) found the two tied on lDDT with empty better on RMSD/TM (mean lDDT
0.966 vs 0.963; RMSD-backbone 3.27 vs 4.20 Å). All self-run models were therefore
run single-sequence — also faster and cheaper. **AlphaFold Server is the
exception**, which is why both AF3 conditions are reported separately rather than
collapsed.

### 4.5 One sample per target, and how AF3's five were reduced

For a fair comparison each method contributes a single prediction not chosen by
reference to the ground truth. Boltz-2, Chai-1 and Protenix were each run to
produce one sample. AlphaFold Server always returns five ranked predictions; we
take the top-ranked (`model_0`).

The effect of this choice was quantified on the first 27 AF3 structures by scoring
all five models per target:

| Selection of AF3's 5 | TM ↑ | lDDT-aa ↑ | RMSD-bb ↓ (Å) |
|---|---|---|---|
| Single random sample (fixed seed) | 0.894 | 0.941 | 2.859 |
| Top confidence (`model_0`) | 0.893 | 0.942 | 2.690 |
| Oracle best-of-5 (vs ground truth) | 0.914 | 0.945 | 2.392 |

**Top-confidence vs random is negligible** (≈0 on TM/lDDT, ~0.17 Å on backbone
RMSD), and both sit well short of the *oracle* best-of-5 — confirming that
confidence selection is **not** answer-cheating. `model_0` was chosen for clean
reproducibility. Note that "best-of-N by a method's own internal score" (ML
confidence, or the deterministic method's energy) is a capability available to all
methods. *(These absolute values predate the current scoring pass; refresh them if
quoted numerically.)*

### 4.6 No structural templates

The 80 targets are deposited PDB entries and AlphaFold Server's template set
extends to 2025-02-03. With templates enabled (the default) AF3 could retrieve the
deposited structure, or a near-identical collagen, as a template — answer leakage.
Templates were disabled (`useStructureTemplate:false`), consistent with the other
models, which use none.

### 4.7 Models considered and excluded

- **RoseTTAFold-All-Atom (RFAA).** Its current release lists modified/unnatural
  amino-acid support as "coming soon"; the only available mechanism (per-residue
  covalent atomisation via SDF) is impractical at collagen's HYP density (tens of
  HYP per structure), so RFAA cannot faithfully model hydroxyproline. Protenix was
  substituted as a second, independently-trained AF3-family datapoint.
- **ESM3 / ESMFold.** Monomer-trained and limited to the 20 canonical amino acids
  (no HYP); a single collagen chain has no meaningful isolated fold. Not suitable
  for an obligate HYP-rich trimer benchmark.

---

## 5. Scoring

A single scorer (`4_scoring/score.py`; US-align + gemmi) scores every predicted
CIF against its experimental reference. Outputs `results/scores_summary.csv` (one
row per target × variant) and `results/scores_per_residue.csv`.

### 5.1 Matching and canonicalization

Parsing canonicalizes residue names (HYP↔PRO treated as equivalent; Amber
protonation/bond variants HID/HIE/HIP, CYX, CYM, ASH, GLH, LYN mapped to standard
letters), uses heavy atoms only, and requires a Cα per residue. Predictions are
matched to references by **sequence correspondence** (a prediction is a
subsequence of the deposited sequence), with the chain assignment chosen to
**maximize matched residues** (geometry breaks ties) — so differing chain IDs and
numbering are handled automatically.

**Two bugs fixed during development**, both of which had been depressing coverage
and scores:

1. **Chain mapping.** `best_chain_map` now maximises matched residues *first*,
   with geometry only as a tie-break. The previous pure min-Cα-RMSD criterion
   mis-paired near-symmetric trimers of unequal length (e.g. 9PUF 20/19/20 →
   coverage 0.98 and wrong RMSD).
2. **Residue naming.** `THREE2ONE` now includes the Amber variants. `tleap`
   renames HIS→HIE, which the parser was silently dropping, truncating sequences
   (6HG7 / 8HHI coverage 0.42 / 0.50). All coverage is now 1.0.

### 5.2 Metrics

- **lDDT, all-atom and backbone** — local Distance Difference Test (Mariani et al.
  2013), superposition-free: for every inter-residue atom pair within R₀ = 15 Å in
  the reference, the fraction of tolerance thresholds {0.5, 1, 2, 4 Å} the model
  reproduces; maximized over chain permutations. "Backbone" restricts atoms to
  N/CA/C/O. **Primary metric.**
- **TM-score** — US-align multimer mode (`-mm 1 -ter 1`), normalized by the
  reference length (`Structure_2`). Secondary; global fold.
- **RMSD, all-atom and backbone** — computed over sequence-matched atoms in
  US-align's global-fit superposition frame (via the `-m` rotation matrix), with
  our own sequence-based residue pairing and **no outlier rejection**.
- **Coverage** — matched reference residues / total.

**Metric complementarity.** lDDT is local and superposition-free, therefore
robust; global RMSD and TM-score are superposition-based and, for these elongated
rods, sensitive to a single mis-registered chain — a structure can be locally
accurate yet globally mis-oriented. **TM-score is largely blind to a one-triplet
register slip** (the optimal superposition slides the alignment) while lDDT and
RMSD expose it. All metrics are therefore reported, with lDDT as primary.

**Comparison basis.** Comparisons are over the **75 structures common to all
methods** (the deterministic builder cannot build 5). The deep-learning models
were additionally run on all 80.

### 5.3 Scoring gotchas

- **`--pdb-id X` overwrites the entire `scores_summary.csv`** with only that
  structure's rows. To update one structure, re-run the full scoring (a few
  minutes); do not use `--pdb-id` for a partial update.
- **Difference from PyMOL RMSD.** Our RMSD differs from PyMOL `rms_cur`/`align`
  because (a) we pair residues by sequence, whereas PyMOL pairs by identical
  chain+resnum — which mismatches, since our generated CIFs renumber per chain
  1..N while experimental files keep deposited numbering; and (b) PyMOL `align`
  performs outlier rejection (`cycles=5`). Our value is the honest
  all-atom-over-all-shared-atoms number.
- **Adding a model** to the scorer is one entry in the `VARIANTS` map plus the
  output-file naming convention `<PDB>_<model>.cif`.

---

## 6. Results

Recomputed from `scores_summary.csv` on 2026-08-15 over the **75 shared targets**
(27 homotrimers, 48 heterotrimers). Deterministic = `fullseq_reregistered_relaxed`.

### 6.1 Primary table (mean / median)

| Metric | Deterministic | Boltz-2 | Chai-1 | Protenix | AF3 (MSA) | AF3 (no MSA) |
|---|---|---|---|---|---|---|
| TM ↑ | 0.863 / 0.879 | **0.896 / 0.919** | 0.890 / 0.920 | 0.883 / 0.906 | 0.889 / 0.916 | 0.863 / 0.904 |
| lDDT all-atom ↑ | 0.887 / 0.912 | **0.946 / 0.961** | 0.936 / 0.955 | 0.929 / 0.946 | 0.937 / 0.957 | 0.926 / 0.952 |
| lDDT backbone ↑ | 0.948 / 0.980 | **0.974 / 0.992** | 0.969 / 0.992 | 0.961 / 0.990 | 0.969 / 0.993 | 0.959 / 0.990 |
| RMSD all-atom ↓ (Å) | 2.365 / 1.580 | 3.056 / 1.255 | 3.030 / 1.315 | 3.054 / 1.341 | **1.998 / 1.193** | 2.960 / 1.422 |
| RMSD backbone ↓ (Å) | 1.862 / 1.215 | 2.590 / 0.968 | 2.540 / 1.066 | 2.605 / 1.067 | **1.675 / 0.901** | 2.580 / 1.089 |

**Mean-vs-median split.** ML models have better *medians* on RMSD (0.90–1.09 Å vs
1.215 Å) but worse *means* — the signature of a heavy failure tail (§6.4). Report
both, or the comparison misleads in either direction. The deterministic method has
the **second-best mean RMSD of all six variants**, behind only AF3 (MSA).

### 6.2 Statistical testing

Paired Wilcoxon signed-rank, deterministic vs each ML variant, n = 75:

| Metric | vs Boltz-2 | vs Chai-1 | vs Protenix | vs AF3 (MSA) | vs AF3 (no MSA) |
|---|---|---|---|---|---|
| **RMSD backbone** | p = 0.98 (n.s.) | p = 0.28 (n.s.) | p = 0.62 (n.s.) | **p = 0.004** | p = 0.58 (n.s.) |
| **RMSD all-atom** | p = 0.64 (n.s.) | p = 0.98 (n.s.) | p = 0.48 (n.s.) | **p = 1×10⁻⁴** | p = 0.64 (n.s.) |
| TM-score | **p = 3×10⁻⁴** | **p = 0.004** | **p = 0.002** | **p = 0.002** | p = 0.85 (n.s.) |
| lDDT all-atom | **p = 3×10⁻¹³** | **p = 2×10⁻¹²** | **p = 3×10⁻¹¹** | **p = 2×10⁻¹²** | **p = 2×10⁻⁹** |
| lDDT backbone | **p = 4×10⁻¹¹** | **p = 2×10⁻⁹** | **p = 4×10⁻⁹** | **p = 5×10⁻¹⁰** | **p = 2×10⁻⁷** |

**Reading.** On **both RMSD measures** the deterministic method is statistically
indistinguishable from Boltz-2, Chai-1, Protenix and AF3 (no MSA) — all p ≥ 0.28.
Only AF3 (MSA) is significantly better. It loses TM-score and both lDDT measures
to every model except AF3 (no MSA), and the lDDT losses are overwhelming
(p ~ 10⁻⁷–10⁻¹³).

### 6.3 Per-target head-to-head (deterministic wins–losses, ties excluded)

| Metric | vs Boltz-2 | vs Chai-1 | vs Protenix | vs AF3 (MSA) | vs AF3 (no MSA) |
|---|---|---|---|---|---|
| RMSD backbone | 31–44 | **41–34** | 36–39 | 25–50 | **38–37** |
| RMSD all-atom | 27–48 | 34–41 | 26–49 | 19–56 | 33–42 |
| TM-score | 20–55 | 31–44 | 26–49 | 28–47 | **38–37** |
| lDDT backbone | 6–69 | 10–65 | 10–65 | 7–68 | 10–65 |
| lDDT all-atom | 2–73 | 6–69 | 5–70 | 3–72 | 6–69 |

The deterministic method wins the **majority** of backbone-RMSD head-to-heads
against Chai-1 and is at parity with Protenix and AF3 (no MSA). It wins only
2–10% of lDDT comparisons — the local-geometry deficit is near-universal, not
outlier-driven.

### 6.4 Robustness and the failure tail

Backbone RMSD, 75 targets:

| Method | > 3 Å (of which hetero) | > 5 Å ("catastrophic") | Worst case | 90th pct |
|---|---|---|---|---|
| **Deterministic** | **14 (12)** | **5** | **6.72 Å** | 4.75 Å |
| Boltz-2 | 22 (19) | 20 | 15.10 Å | 5.86 Å |
| Chai-1 | 23 (19) | 20 | 10.86 Å | 5.84 Å |
| Protenix | 24 (19) | 18 | 12.36 Å | 5.80 Å |
| AF3 (MSA) | 13 (11) | 7 | 7.25 Å | **4.73 Å** |
| AF3 (no MSA) | 20 (14) | 14 | 27.60 Å | 5.52 Å |

**This is the strongest result for the deterministic method.** It produces **5
catastrophic failures against 18–20** for Boltz-2, Chai-1 and Protenix — a ~4×
reduction — and its worst case (6.7 Å) is roughly half theirs. Only AF3 (MSA) is
comparable. Physics degrades gracefully; the diffusion models occasionally fail
outright.

**Every method's failures concentrate in heterotrimers** (11–19 of each method's
>3 Å failures). Chain register and global placement in heterotrimers is an open
problem for the field, not a quirk of one method.

### 6.5 Homotrimer vs heterotrimer stratification

| Method | RMSD-bb homo (27) | RMSD-bb hetero (48) | lDDT-aa homo | lDDT-aa hetero |
|---|---|---|---|---|
| Deterministic | 1.360 | 2.144 | 0.915 | 0.871 |
| Boltz-2 | 1.664 | 3.111 | 0.959 | 0.938 |
| Chai-1 | 2.041 | 2.820 | 0.951 | 0.927 |
| Protenix | 1.709 | 3.109 | 0.948 | 0.918 |
| AF3 (MSA) | **1.311** | **1.880** | 0.952 | 0.929 |
| AF3 (no MSA) | 2.063 | 2.871 | 0.949 | 0.913 |

**On homotrimers the deterministic method (1.360 Å) beats every model except AF3
(MSA)**, and is within 0.05 Å of it — near-parity with the frontier on the simpler
class. All methods degrade on heterotrimers. The deterministic method's lDDT-aa
drop is the largest (0.915 → 0.871, −0.044 vs ML's −0.020 to −0.036):
heterotrimer sequence diversity brings more non-Gly/Pro/Hyp side chains,
compounding its side-chain weakness.

### 6.6 The side-chain deficit

Mean lDDT backbone − all-atom:

| Method | lDDT-bb | lDDT-aa | **Δ (side-chain penalty)** |
|---|---|---|---|
| **Deterministic** | 0.948 | 0.887 | **0.061** |
| Boltz-2 | 0.974 | 0.946 | 0.028 |
| Chai-1 | 0.969 | 0.936 | 0.033 |
| Protenix | 0.961 | 0.929 | 0.033 |
| AF3 (MSA) | 0.969 | 0.937 | 0.031 |
| AF3 (no MSA) | 0.959 | 0.926 | 0.033 |

The deterministic penalty is **~2× that of every ML model**. The fold is right;
the side chains are not placed as well. This localises the deficit to a specific,
addressable module (rotamer/side-chain optimisation) rather than the core geometry.

**Per-atom decomposition.** A superposition-free per-atom lDDT breakdown by
residue and atom class gives the deterministic-vs-ML gap:

| Residue | Atom | determ lDDT | ML lDDT | gap |
|---|---|---|---|---|
| GLY | backbone | 0.741 | 0.787 | +0.047 |
| PRO | backbone | 0.739 | 0.779 | +0.040 |
| PRO | side chain | 0.712 | 0.769 | +0.057 |
| HYP | backbone | 0.747 | 0.780 | +0.033 |
| HYP | side chain | 0.731 | 0.770 | +0.039 |
| OTHER | backbone | 0.700 | 0.775 | +0.075 |
| **OTHER** | **side chain** | **0.564** | **0.699** | **+0.135** |

*(These per-atom values omit the chain-permutation maximization used for the
headline lDDT, so their absolute scale is depressed and not comparable to 0.887;
they are a relative breakdown computed identically for both methods.)*

Findings: (i) the largest, most-recoverable loss is on **non-Gly/Pro/Hyp side
chains** — the charged/polar residues in Gly-X-Y interruptions, a rotamer-placement
problem; (ii) **Pro/Hyp side chains are only modestly behind**, ruling out
ring-pucker as the main cause; (iii) the backbone also lags, worst at the
interruptions. Atom composition of the dataset: GLY 20.7%, PRO 35.1%, HYP 24.9%,
OTHER 19.3%. Weighting by atom count, ~25% of the total lDDT deficit is
interruption-region side chains and ~50% is backbone.

### 6.7 Per-residue positional analysis

Mean per-residue lDDT; terminal = outer 12% of each chain:

- **The deterministic gap is uniform along the chain** — ~0.045 in the core,
  ~0.040 at the termini. It is a constant side-chain offset, *not* a localised
  breakdown at any region.
- **All methods fray at the termini** (backbone lDDT drops 0.022–0.033). The
  deterministic builder frays somewhat more on the backbone but less on all-atom,
  simply because its all-atom score is already uniformly depressed.

### 6.8 CDSM pipeline staging (ablation)

Mean over the 75 buildable targets:

| Stage | lDDT-aa | lDDT-bb | RMSD-bb (Å) | TM |
|---|---|---|---|---|
| Core only | 0.884 | **0.964** | 1.924 | 0.799 |
| Full sequence | 0.843 | 0.915 | 3.020 | 0.859 |
| + Re-registered | 0.870 | 0.944 | 1.975 | 0.863 |
| + Relaxed (reported) | **0.887** | 0.948 | **1.862** | **0.863** |

- **Re-registration is the single largest gain**: backbone RMSD 3.02 → 1.98 Å
  (−35%), confirming that chain register is the dominant error mode in the naive
  build.
- **Relaxation buys all-atom quality**: lDDT-aa 0.870 → 0.887 with little backbone
  change — it is doing side-chain work, consistent with §6.6.
- Core-only achieves the best backbone lDDT (0.964) but the worst TM (0.799)
  because it models less of the structure. Core→Full-seq raises TM (trimmed
  residues added back over the full reference length) while lDDT/RMSD worsen
  (full-seq includes the mis-registered chains); re-registration recovers them.

A fifth stage group, **Native (n=7)** — the subset THeBuScr builds without any
reframing (`3B0S 1YM8 4DMT 3P46 5Y46 3POD 7VEG`) — is drawn in `fig_cdsm_*` and
sits highest on lDDT/RMSD, being the cleanest subset.

### 6.9 Case studies

**A. All ML models fail, deterministic succeeds** (RMSD-bb):

| Target | Type | Deterministic | Boltz-2 | Chai-1 | Protenix | AF3 (MSA) |
|---|---|---|---|---|---|---|
| **1WZB** | hetero | **0.47 Å** (lDDT-aa 0.985) | 5.7 | 5.4 | 5.5 | 5.5 |
| **4Z1R** | hetero | **1.15 Å** | 5.6 | 5.6 | 5.7 | 5.8 |
| **1YM8** | homo | **0.49 Å** | 8.8 | 9.0 | 5.2 | 0.8 |
| **2DRX** | hetero | **1.13 Å** | 5.5 | 5.5 | 5.5 | 1.0 |

1WZB and 4Z1R are the headline cases: every deep-learning model places the chains
~5.5 Å off while the physics builder is sub-1.2 Å. In 1WZB the ML models still
score lDDT ≈ 0.98 — locally perfect, globally misplaced. This is the cleanest
demonstration that lDDT and RMSD measure different things, and that the ML failure
mode is *global register*.

**B. Shared hard cases** (all methods 4.4–6.9 Å): 3AH9, 6A0C, 8TW0, 8ZMM, 8ZMV —
all heterotrimers. 8TW0 is the deterministic method's worst structural failure
(lDDT-aa 0.529).

**C. Counter-examples — deterministic fails where ML succeeds** (report for
balance): 3U29 (homo) 5.25 Å vs 0.6–0.8; 9J1T (hetero) 4.66 Å vs 0.7–1.0
(except Chai-1, 5.7).

**D. Showcase targets where every method is excellent** (RMSD-bb Å / lDDT-aa):
3B0S homo 0.38/0.976; 1K6F homo 0.34/0.973; 2CUO hetero 0.38/0.965. 3B0S has the
smallest all-atom lDDT gap of any target. **Caveat:** all of these are
Gly/Pro/Hyp-only sequences — no target with diverse side chains reaches this bar,
itself consistent with §6.6.

### 6.10 Cost and throughput

| Method | Compute | ~Time/structure | ~Cost/structure |
|---|---|---|---|
| **Deterministic** | CPU (laptop) | seconds | **$0** |
| AlphaFold3 | AlphaFold Server | — | $0 (capped 30 jobs/day, manual) |
| Chai-1 | Modal L40S | 30–60 s | ~$0.02–0.03 |
| Boltz-2 | hosted API | — | ~$0.025 |
| Protenix | Modal L40S | 1–2 min | ~$0.05–0.08 |

ML costs are estimates (only Boltz-2's is a published price) and exclude one-time
setup. The full 80-target benchmark costs a few dollars for any ML model — cost
does not separate them. What separates the deterministic method is the *category*:
no GPU, no weights, no per-structure quota, no network dependency, and exactly
reproducible output.

---

## 7. Limitations and caveats

1. **AF3 (MSA) was not run under matched conditions.** AlphaFold Server forces its
   own MSA (the other models were single-sequence) and returns 5 ranked
   predictions (we took the top-ranked). The confidence-selection effect was
   measured to be small (§4.5), but the MSA advantage is uncontrolled. The
   `af3_nomsa` condition is the closer comparison; note that its results are
   markedly worse and closer to the deterministic method on every metric.
2. **Deterministic coverage.** The builder fails on 5 of 80 targets; all
   cross-method comparisons are over the 75 it can build, so its 6% failure-to-
   build rate is *not* penalised in the score tables and must be reported
   separately. The ML models predict all 80.
3. **Single-sample sampling.** Boltz-2, Chai-1 and Protenix each produced one
   sample per target (no selection); the deterministic method is likewise a single
   deterministic output. Neither side uses ensembling.
4. **No structural templates** were used by any method (AF3 templates explicitly
   disabled to prevent leakage of the deposited structure).
5. **Metric caveat.** A single bad reference can dominate RMSD means — see §9.2.

---

## 8. Planned improvements (not implemented)

- **Statistical-library side-chain repacking** (Dunbrack backbone-dependent
  library / SCWRL-style) restricted to the standard, non-Gly/Pro/Hyp interruption
  residues — the one side-chain lever that escapes the force-field ceiling, since
  it scores rotamers by observed-structure probability rather than energy. Bounded
  upside (~25% of the gap).
- **Backbone refinement at interruptions** — the idealized THeBuScr helix is
  geometrically off where the Gly-X-Y repeat breaks (~50% of the gap); untouched
  by any relaxation.
- **Building the 5 interrupted structures** — guard the propensity divide-by-zero,
  or reframe piecewise across interruptions.
- **A controlled local AF3 run** (own weights, single-sequence, one sample) to
  resolve the one uncontrolled variable behind AF3 (MSA)'s win.

---

## 9. Provenance and reproducibility

### 9.1 Scoring generations

Results have been recomputed several times. Numbers in the superseded notes are
**not** interchangeable with those in §6:

| Generation | Distinguishing feature |
|---|---|
| Pre-8ZMO-fix | Deterministic RMSD-bb mean 2.251 Å; appears in the old `PAPER_METHODS.md` §6 and `METHODS_ml_benchmark.md` §7 |
| Post-8ZMO-fix | Deterministic RMSD-bb 1.862 Å; 620 rows; appears in `RESULTS_ml_benchmark.md` |
| **Current (2026-08-15)** | **700 rows; `af3` split into `af3_msa`/`af3_nomsa`; Boltz-2 structures regenerated** |

The Boltz-2 regeneration (2026-08-14) changed its tail materially: worst-case
backbone RMSD 9.00 → 15.10 Å and catastrophic failures 13 → 20, while its mean
lDDT barely moved. Any Boltz-2 number from an earlier generation is stale.

### 9.2 The 8ZMO reference correction

A single corrected experimental reference accounts for a ~0.39 Å improvement in
every method's mean backbone RMSD. Working backwards from each method's mean shift
recovers 8ZMO's old value independently five times over, at ~30.2–30.4 Å (the same
answer from all-atom RMSD, an independent metric). Its contribution to the mean was
30.2/75 = 0.40 Å on its own.

**Why lDDT barely moved:** 8ZMO's old lDDT-aa was ≈0.77, now ≈0.98 — a gain of
0.21/75 = **0.003** on the mean, exactly what was observed. An lDDT of 0.77
alongside a 30 Å RMSD is the fingerprint of a **globally misplaced but locally
correct** reference.

Two consequences: the correction is **method-neutral** (it moved all methods almost
identically, so no comparative conclusion depends on it); and it is a concrete,
quantified illustration of the metric complementarity argued in §5.2 — one bad
reference perturbed RMSD by 0.39 Å and lDDT by 0.003.

### 9.3 Environments

- **Deterministic build, relaxation, tleap:** `MIT_environment` conda env (has
  `tleap` on PATH, `parmed`, `openmm` 8.4):
  `PATH="/opt/anaconda3/envs/MIT_environment/bin:$PATH" /opt/anaconda3/envs/MIT_environment/bin/python ...`
- **Scoring and figures:** base `/opt/anaconda3/bin/python3.12` (`gemmi`, `numpy`,
  `pandas`, `matplotlib`, `scipy`; no parmed/openmm/tleap). US-align at
  `tools/USalign`.

### 9.4 Model-specific reproducibility notes

- Deep-learning drivers follow one pattern per model: a Modal app defining the GPU
  `fold` function, and a driver that builds the model-specific input from the
  shared target list, runs it, and writes `<PDB>_<model>.cif`. Weights are cached
  in Modal Volumes. Long batches are run under `caffeinate`.
- **Protenix specifics:** CUDA *devel* base image (nvcc needed for a JIT-compiled
  fused-LayerNorm kernel; `CUDA_HOME` set); `clang`/`build-essential` installed (a
  dependency builds C extensions from source); the 1.48 GB checkpoint is on a slow
  CDN and was fetched with an explicit size-checked download to avoid truncation;
  `cuequivariance` kernels compile (~10–15 min) on the first fold, then cache.
- **AlphaFold Server outputs** (per-job folders of `fold_<id>_model_0..4.cif` +
  `summary_confidences_*.json`) are ingested by selecting `model_0` per job and
  copying to `<PDB>_af3_msa.cif`.

### 9.5 Commands

```bash
# Re-run download and filtering
python 1_download/download_collagen.py

# Build all deterministic variants
python 2_deterministic_build/run_pipeline.py --all

# Score everything (do NOT use --pdb-id for partial updates; see §5.3)
python 4_scoring/score.py

# Regenerate figures
python figures/make_figures.py

# Publish data + dataset card to HuggingFace
python 1_download/upload_to_huggingface.py --all --trajectories
```

Results may vary slightly between runs as the PDB index is updated. The dataset
described here reflects a query run on **2026-07-14**, returning 758 candidates
and yielding **80 passing structures**.

---

## 10. Key parameters (quick reference)

| Component | Setting |
|---|---|
| RCSB search | keyword collagen/triple-helix; chain count ∈ {3,6,9,12}; protein-only |
| Filters | 3 chains; residues ∈ 20 AA + HYP; length 10–200; Gly ≥ 0.25; internal gap ≤ 5 |
| Dedup | exact chain-sequence multiset; keep min resolution, then min ASU chains, then min PDB ID |
| Sequences | from ATOM records; `O` = HYP |
| THeBuScr | byte-exact port; core vs extend completion |
| Terminal extension | Kabsch screw extrapolation from the adjacent pair (§3.2) |
| Register fix | whole-triplet shifts; minimize axial terminal spread; ≥ 2.0 Å threshold |
| Side chains | tleap ff14SB; O→HYP; HYP CCD bond topology injected |
| Minimization | OpenMM 8.4; OBC2; NoCutoff; HBonds; backbone restraint k = 1000 kJ/mol/nm²; continuous minimizeEnergy (≤ 2000 iter) |
| Boltz-2 | `boltz-2.1`; empty MSA; HYP = ccd modification, 0-indexed; ~$0.025/structure |
| Chai-1 | `chai_lab` 0.6.1; 3 recycles; 200 diffusion steps; 1 sample; inline `(HYP)`; seed 42 |
| Protenix | `protenix_base_default_v1.0.0`; 10 recycles; 200 steps; `--use_msa false`; 1-indexed ptmPosition |
| AlphaFold3 | AlphaFold Server; templates disabled; `model_0` of 5; MSA forced (`af3_msa`) and no-MSA (`af3_nomsa`) conditions |
| lDDT | R₀ = 15 Å; thresholds {0.5, 1, 2, 4 Å}; chain-permutation max |
| TM-score | US-align `-mm 1 -ter 1`, reference-normalized |
| RMSD | sequence-matched atoms, US-align frame, no outlier rejection |
| Dataset | 80 structures (30 homo-, 50 heterotrimer); 75 shared for comparison |
