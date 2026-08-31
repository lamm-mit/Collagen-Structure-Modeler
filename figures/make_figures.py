#!/usr/bin/env python3
"""
make_figures.py — publication figures for the collagen structure-prediction benchmark.

Reads 4_scoring/results/{scores_summary.csv, scores_per_residue.csv} and the manifest,
writes each figure as PNG (300 dpi) and PDF into figures_png/ and figures_pdf/.
Comparisons use the 75 structures
common to all methods (deterministic builds only 75/80).

Figures:
  fig1_distributions        box+points per metric across methods
  fig2_sidechain_gap        backbone vs all-atom lDDT (the side-chain deficit)
  fig3_headtohead_scatter   per-target deterministic-vs-ML scatter (lDDT-aa, RMSD-bb)
  fig4_ecdf_rmsd_bb         cumulative distribution of backbone RMSD
  fig6_positional_lddt      mean per-residue lDDT along the chain (N->C)
  fig_winrate_heatmap       % of targets deterministic wins, metric x ML method
  fig_winmargin_rmsd_bb     paired per-target RMSD-bb difference (determ - ML) + Wilcoxon

CDSM pipeline-stage figures (deterministic builder only, no ML):
  fig_cdsm_tmscore          TM-score across the 5 build stages
  fig_cdsm_lddt             lDDT backbone + all-atom across the 5 build stages
  fig_cdsm_rmsd             RMSD backbone + all-atom across the 5 build stages
  (+ 4_scoring/results/cdsm_stage_table.csv, mean/median per stage x metric)
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data_locations import experimental_cif_dir, manifest_path, scores_dir  # noqa: E402

RES = scores_dir()
MANIFEST = manifest_path("csv")

DET = "fullseq_reregistered_relaxed"
ORDER = [DET, "boltz", "chai", "protenix", "af3_nomsa"]
ML = ["boltz", "chai", "protenix", "af3_nomsa"]
LABEL = {DET: "Deterministic", "boltz": "Boltz-2", "chai": "Chai-1",
         "protenix": "Protenix", "af3_nomsa": "AlphaFold3"}
# af3_nomsa is the only AlphaFold3 condition plotted, so it keeps AF3's original
# colour, dash pattern and plain "AlphaFold3" label; the MSA condition (af3_msa)
# is still scored into scores_summary.csv but is not shown in any figure.
COLOR = {DET: "#D55E00", "boltz": "#0072B2", "chai": "#009E73",
         "protenix": "#CC79A7", "af3_nomsa": "#E69F00"}
LS = {DET: "-", "boltz": (0, (5, 1)), "chai": (0, (1, 1)),
      "protenix": (0, (3, 1, 1, 1)), "af3_nomsa": (0, (4, 1, 1, 1, 1, 1))}

# ── CDSM pipeline stages (deterministic builder only) ────────────────────────
# THeBuScr's Phase-1 propensity math divides by zero on these (internal Gly-X-Y
# register interruptions), so they have no deterministic build at any stage.
KNOWN_ZERODIV = {"1EI8", "6M80", "5K86", "7LXQ", "7LXP"}

# (stage label, scores_summary variant, restrict to the "native" subset?)
CDSM_STAGES = [
    ("Native",        "coreonly",                     True),
    ("Core only",     "coreonly",                     False),
    ("Full seq",      "fullseq",                      False),
    ("Re-registered", "fullseq_reregistered",         False),
    ("Relaxed",       "fullseq_reregistered_relaxed", False),
]
CDSM_METRICS = [("tm_score", "TM-score"),
                ("global_lddt_backbone", "lDDT (bb)"),
                ("global_lddt_allatom", "lDDT (aa)"),
                ("global_rmsd_backbone", "RMSD bb (Å)"),
                ("global_rmsd_allatom", "RMSD aa (Å)")]
CDSM_FIGSIZE = (7.8, 4.6)          # identical across all three stage charts
STAGE_C = "#2C3E50"                # single-series (TM) colour
BB_C, AA_C = "#4B4B4B", "#D55E00"  # backbone / all-atom — matches fig2_sidechain_gap

# (column, pretty, arrow, higher_is_better)
METRICS = [
    ("tm_score", "TM-score", "↑", True),
    ("global_lddt_allatom", "lDDT (all-atom)", "↑", True),
    ("global_lddt_backbone", "lDDT (backbone)", "↑", True),
    ("global_rmsd_allatom", "RMSD all-atom (Å)", "↓", False),
    ("global_rmsd_backbone", "RMSD backbone (Å)", "↓", False),
]

# ── style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
})
INK = "#222222"


# ── paper panel geometry ─────────────────────────────────────────────────────
# fig2b, fig4 and fig6b are the three panels of the paper figure. They must be
# byte-identical in size and share an axes rectangle so they line up when set
# side by side, so they use a fixed subplots_adjust and savefig.bbox=None
# ("tight" would crop each one to its own content and break the alignment).
PAPER_FIGSIZE = (2.8, 3.0)
PAPER_RC = {"font.size": 5.5, "axes.titlesize": 6.5, "axes.labelsize": 5.5,
            "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 4.6,
            "axes.linewidth": 0.6, "grid.linewidth": 0.4,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.major.size": 2, "ytick.major.size": 2,
            "savefig.bbox": None}
PAPER_ADJUST = dict(left=0.145, right=0.98, top=0.925, bottom=0.115)

# Figures used in the paper are written as PNG and PDF into figures_png/ and
# figures_pdf/, under the manuscript's own numbering. Everything
# else is exploratory and goes to other_figs/ as PNG only, renamed to a plain
# "fig_" stem. To promote a figure, add it to PAPER_FIGURES with its paper name.
FORMATS = ("png", "pdf")
OTHER_DIR = "other_figs"

PAPER_FIGURES = {
    "Figure1_chain_length":       "fig_1_a_chain_length",
    "Figure3_top_triplets":       "fig_1_b_GXYtriplets",
    "Figure2_GXY_composition":    "fig_1_c_GXYcomposition",
    "Figure4_identity_heatmap":   "fig_1_d_sequenceidentity",
    "fig_cdsm_lddt":              "fig_5_a_cdsm_lddt",
    "fig_cdsm_rmsd":              "fig_5_b_cdsm_rmsd",
    "fig_cdsm_tmscore":           "fig_5_c_cdsm_tmscore",
    "fig_bench_tmscore":          "fig_6_d_bench_tmscore",
    "fig_bench_combined":         "fig_6_d-f_bench_combined",
    "fig_bench_rmsd":             "fig_6_e_bench_rmsd",
    "fig_bench_lddt":             "fig_6_f_bench_lddt",
    "fig4_ecdf_rmsd_bb":          "fig_7_a_ecdf_rmsd_bb",
    "fig6b_positional_rmsd_bb":   "fig_7_b_positional_rmsd_bb",
    "fig2b_sidechain_gap_rmsd":   "fig_7_c_sidechain_gap_rmsd",
    "Figure6_triplet_vocabulary": "fig_8_a_triplet_vocabulary",
    "fig_cutoff_grid":            "fig_8_b_cutoff_grid",
}


def _other_name(name):
    """Strip any figure numbering, leaving a plain fig_<topic> stem."""
    m = re.match(r"^(?:Figure|fig)\d*[a-z]?_(.*)$", name)
    stem = m.group(1) if m else name
    return stem if stem.startswith("fig_") else f"fig_{stem}"


def save(fig, name):
    if name in PAPER_FIGURES:
        out = PAPER_FIGURES[name]
        for ext in FORMATS:
            d = os.path.join(HERE, f"figures_{ext}")
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, f"{out}.{ext}"))
        note = f"{out}." + "/.".join(FORMATS)
    else:
        out = _other_name(name)
        d = os.path.join(HERE, OTHER_DIR)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{out}.png"))
        note = f"{OTHER_DIR}/{out}.png"
    plt.close(fig)
    print(f"  wrote {note}")


def _light(hexc, f=0.55):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# ── data ─────────────────────────────────────────────────────────────────────
def load():
    s = pd.read_csv(os.path.join(RES, "scores_summary.csv"))
    man = pd.read_csv(MANIFEST)[["pdb_id", "kind"]]
    man["pdb_id"] = man["pdb_id"].str.upper()
    kind = dict(zip(man.pdb_id, man.kind))
    # shared set = pdbs present for every variant in ORDER
    counts = s[s.variant.isin(ORDER)].groupby("pdb_id").variant.nunique()
    shared = set(counts[counts == len(ORDER)].index)
    s = s[s.pdb_id.isin(shared)].copy()
    s["kind"] = s.pdb_id.map(kind)
    # wide: {variant: {metric: series indexed by pdb}}
    piv = {v: s[s.variant == v].set_index("pdb_id") for v in ORDER}
    pdbs = sorted(shared)
    return s, piv, pdbs, kind


def col(piv, variant, metric, pdbs):
    return piv[variant].loc[pdbs, metric].to_numpy()


# ═══ dataset characterisation (Figures 1-4) ══════════════════════════════════
# These describe the benchmark set itself, not any method, so they read only the
# manifest. Counting conventions match the original hand-made versions: chains
# are counted per trimer (a homotrimer's sequence counts three times) and only
# Gly-initiated triplets are tallied, giving 1,881 Gly-X-Y triplets over 80
# structures. Rebuilt here so they regenerate from the current manifest.

# Figures 1-3 share a figure size so they present at a common aspect ratio.
DATASET_FIGSIZE = (7.2, 4.2)
DATASET_HEATMAP_FIGSIZE = (7.5, 5.6)   # fig 4: slightly wider than tall
DATASET_SQUARE_FIGSIZE = (6.4, 6.4)    # fig 6: vocabulary vs deposition rate
VOCAB_C, VOCAB_BAR_C = "#D55E00", "#A8CDE4"
VOCAB_BAR_AXIS_C = "#3E7EA6"
# Stated training-data cutoffs of the learned methods.
MODEL_CUTOFFS = {"Chai-1": "2021-01-12", "AF3 / Protenix": "2021-09-30",
                 "Boltz-2": "2023-06-01"}
# Chai-1's label sits left of its line, the other two right, so the two 2021
# cutoffs (8 months apart) do not collide.
CUTOFF_LABEL_LEFT = {"Chai-1"}
# Larger than the benchmark figures: these print at full width in the paper.
DATASET_RC = {"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
              "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
              "legend.fontsize": 10.5}
# Figure 2 keeps residues reaching this combined (X + Y) frequency. There is a
# natural break in the distribution here: Phe is 2.66% and the next residue,
# Val, is 1.28%, so the cutoff keeps P O K R D E A L Q F.
GXY_MIN_PCT = 2.5
KIND_C = {"homotrimer": "#0072B2", "heterotrimer": "#56B4E9"}
POS_C = {"X": "#0072B2", "Y": "#56B4E9"}
TRIPLET_C, TRIPLET_OTHER_C = "#0072B2", "#0D0F51"
N_TOP_TRIPLETS = 11


def _reframe(seq):
    """Trim a chain to Gly-X-Y register (see step1_backbone_builder.reframe)."""
    s = str(seq).upper().strip().replace("﻿", "")
    f = min(range(3), key=lambda k: sum(1 for i in range(k, len(s), 3) if s[i] != "G"))
    s = s[f:]
    return s[: len(s) - (len(s) % 3)]


def dataset_chains(man):
    """[(pdb_id, kind, sequence), ...] one row per chain; homotrimers count 3x."""
    out = []
    for _, r in man.iterrows():
        pid, kind = str(r["pdb_id"]).upper(), str(r["kind"]).strip().lower()
        seqs = ([r["chain_a_sequence"]] * 3 if kind == "homotrimer" else
                [r["chain_a_sequence"], r["chain_b_sequence"], r["chain_c_sequence"]])
        out += [(pid, kind, str(s).upper().strip()) for s in seqs]
    return out


def fig_dataset_chain_length(man):
  with mpl.rc_context(DATASET_RC):
    fig, ax = plt.subplots(figsize=DATASET_FIGSIZE)
    lens = {k: [] for k in KIND_C}
    for _, r in man.iterrows():
        lens[str(r["kind"]).strip().lower()].append(
            np.mean([r["len_a"], r["len_b"], r["len_c"]]))
    bins = np.arange(np.floor(min(sum(lens.values(), []))) - 0.5,
                     np.ceil(max(sum(lens.values(), []))) + 1.5, 1.0)
    order = ["heterotrimer", "homotrimer"]
    ax.hist([lens[k] for k in order], bins=bins, stacked=True, rwidth=0.82,
            color=[KIND_C[k] for k in order],
            label=[f"{k.capitalize()} (n={len(lens[k])})" for k in order],
            edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("Mean chain length (residues)")
    ax.set_ylabel("Number of structures")
    ax.set_title(f"Chain-length distribution ({len(man)} structures)")
    ax.grid(axis="y", zorder=0)
    ax.legend()
    fig.tight_layout()
    save(fig, "Figure1_chain_length")


def fig_dataset_gxy_composition(man):
    """Residue frequency at the X and Y positions, as % of all Gly-X-Y triplets."""
    from collections import Counter
    cx, cy, total = Counter(), Counter(), 0
    for _, _kind, seq in dataset_chains(man):
        rs = _reframe(seq)
        for i in range(0, len(rs), 3):
            t = rs[i:i + 3]
            if t.startswith("G"):
                cx[t[1]] += 1; cy[t[2]] += 1; total += 1
    letters = sorted(set(cx) | set(cy), key=lambda a: -(cx[a] + cy[a]))
    kept = [a for a in letters if 100 * (cx[a] + cy[a]) / total >= GXY_MIN_PCT]
    dropped = len(letters) - len(kept)
    letters = kept
    x = np.arange(len(letters))
    ctx = mpl.rc_context(DATASET_RC); ctx.__enter__()
    fig, ax = plt.subplots(figsize=DATASET_FIGSIZE)
    ax.bar(x - 0.2, [100 * cx[a] / total for a in letters], 0.4,
           color=POS_C["X"], label="X position", zorder=3)
    ax.bar(x + 0.2, [100 * cy[a] / total for a in letters], 0.4,
           color=POS_C["Y"], label="Y position", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(letters)
    ax.set_xlabel("Residue (single-letter code; O = hydroxyproline)")
    ax.set_ylabel("Frequency (% of all triplets)")
    ax.set_title(f"Composition of the X and Y positions ({total:,} Gly-X-Y triplets)")
    ax.text(0.99, 0.72, f"{dropped} residues below {GXY_MIN_PCT}% combined "
            f"frequency not shown", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#888888")
    ax.grid(axis="y", zorder=0)
    ax.legend()
    fig.tight_layout()
    save(fig, "Figure2_GXY_composition")
    ctx.__exit__(None, None, None)


def fig_dataset_top_triplets(man):
    """Most frequent Gly-X-Y triplets. The 'Other' bar pools every remaining
    triplet, including structure-unique ones, so the bars sum to 100%."""
    from collections import Counter, defaultdict
    cnt, structs = Counter(), defaultdict(set)
    for pid, _kind, seq in dataset_chains(man):
        rs = _reframe(seq)
        for i in range(0, len(rs), 3):
            t = rs[i:i + 3]
            if t.startswith("G"):
                cnt[t] += 1; structs[t].add(pid)
    total = sum(cnt.values())
    top = cnt.most_common(N_TOP_TRIPLETS)
    rest = {t: c for t, c in cnt.items() if t not in dict(top)}
    labels = [t for t, _ in top] + ["Other"]
    vals = [c for _, c in top] + [sum(rest.values())]
    notes = [f"{100*c/total:.1f}% ({c} occ./{len(structs[t])} struct.)" for t, c in top]
    notes += [f"{100*vals[-1]/total:.1f}% ({vals[-1]} occ./{len(rest)} triplets)"]
    colors = [TRIPLET_C] * len(top) + [TRIPLET_OTHER_C]

    y = np.arange(len(labels))
    xmax = 50.0
    ctx = mpl.rc_context(DATASET_RC); ctx.__enter__()
    fig, ax = plt.subplots(figsize=DATASET_FIGSIZE)
    ax.barh(y, [100 * v / total for v in vals], color=colors, zorder=3)
    for yy, v, n in zip(y, vals, notes):
        pct = 100 * v / total
        if pct > 0.6 * xmax:      # label would overrun the trimmed axis
            ax.text(pct - 1.0, yy, n, va="center", ha="right", fontsize=9,
                    color="white", fontweight="bold", zorder=4)
        else:
            ax.text(pct + 1.0, yy, n, va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(f"Frequency (% of all {total:,} Gly-X-Y triplets)")
    ax.set_title("Most frequent Gly-X-Y triplets")
    ax.grid(axis="x", zorder=0)
    fig.tight_layout()
    save(fig, "Figure3_top_triplets")
    ctx.__exit__(None, None, None)


def _needleman_wunsch_identity(a, b, match=1, mismatch=-1, gap=-1):
    """% identity of a global alignment, normalised by the full alignment length
    (not by the shorter sequence, which would ignore unmatched overhangs)."""
    n, m = len(a), len(b)
    F = np.zeros((n + 1, m + 1))
    F[:, 0] = np.arange(n + 1) * gap
    F[0, :] = np.arange(m + 1) * gap
    P = np.zeros((n + 1, m + 1), dtype=np.int8)
    P[:, 0] = 1; P[0, :] = 2; P[0, 0] = 0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = F[i - 1, j - 1] + (match if ai == b[j - 1] else mismatch)
            up, left = F[i - 1, j] + gap, F[i, j - 1] + gap
            if diag >= up and diag >= left:
                F[i, j], P[i, j] = diag, 0
            elif up >= left:
                F[i, j], P[i, j] = up, 1
            else:
                F[i, j], P[i, j] = left, 2
    i, j, matches, aln_len = n, m, 0, 0
    while i > 0 or j > 0:
        k = P[i, j]
        if k == 0:
            matches += a[i - 1] == b[j - 1]; i -= 1; j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
        aln_len += 1
    return 100.0 * matches / aln_len


def fig_dataset_identity_heatmap(man):
    """Pairwise sequence identity between structures (chain A), hierarchically
    clustered. Identity is over the full global-alignment length, so terminal
    overhangs count against the score and the range extends well below 50%."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    seqs = [str(s).upper().strip() for s in man.chain_a_sequence]
    n = len(seqs)
    ident = np.eye(n) * 100.0
    for i in range(n):
        for j in range(i + 1, n):
            ident[i, j] = ident[j, i] = _needleman_wunsch_identity(seqs[i], seqs[j])
    order = leaves_list(linkage(squareform(100.0 - ident, checks=False), "average"))
    ctx = mpl.rc_context(DATASET_RC); ctx.__enter__()
    fig, ax = plt.subplots(figsize=DATASET_HEATMAP_FIGSIZE)
    vmin = float(np.floor(ident[~np.eye(n, dtype=bool)].min() / 10.0) * 10)
    im = ax.imshow(ident[np.ix_(order, order)], cmap="Blues", vmin=vmin, vmax=100,
                   aspect="auto")   # fill the taller axes instead of forcing square
    ax.set_xlabel(f"{n} structures (clustered)")
    ax.set_ylabel(f"{n} structures (clustered)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Pairwise sequence identity")
    fig.colorbar(im, ax=ax, label="% identity", fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "Figure4_identity_heatmap")
    ctx.__exit__(None, None, None)


def fig_dataset_deposition_year(man):
    """When the benchmark structures entered the PDB."""
    ctx = mpl.rc_context(DATASET_RC); ctx.__enter__()
    years = pd.to_datetime(man["deposition_date"]).dt.year
    order = ["heterotrimer", "homotrimer"]
    by = {k: years[man["kind"].str.strip().str.lower() == k] for k in order}
    bins = np.arange(years.min() - 0.5, years.max() + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=DATASET_FIGSIZE)
    ax.hist([by[k] for k in order], bins=bins, stacked=True, rwidth=0.82,
            color=[KIND_C[k] for k in order],
            label=[f"{k.capitalize()} (n={len(by[k])})" for k in order],
            edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("Year deposited in the PDB")
    ax.set_ylabel("Number of structures")
    ax.set_title(f"Deposition dates ({len(man)} structures, "
                 f"{years.min()}\u2013{years.max()})")
    ax.grid(axis="y", zorder=0)
    ax.legend()
    fig.tight_layout()
    save(fig, "Figure5_deposition_year")
    ctx.__exit__(None, None, None)


def fig_dataset_triplet_vocabulary(man):
    """Cumulative distinct Gly-X-Y triplets against deposition date, over a
    histogram of deposition rate. Shows that the benchmark's sequence vocabulary
    is still expanding, and where each model's training cutoff falls."""
    ctx = mpl.rc_context(DATASET_RC); ctx.__enter__()
    d = man.assign(d=pd.to_datetime(man.deposition_date)).sort_values("d")
    yr = d.d.dt.year + (d.d.dt.dayofyear - 1) / 365.25

    seen, xs, ys = set(), [], []
    for (_, r), y in zip(d.iterrows(), yr):
        rs = _reframe(str(r.chain_a_sequence))
        seqs = ([r.chain_a_sequence] * 3
                if str(r["kind"]).strip().lower() == "homotrimer"
                else [r.chain_a_sequence, r.chain_b_sequence, r.chain_c_sequence])
        for s in seqs:
            rs = _reframe(str(s))
            seen |= {rs[i:i+3] for i in range(0, len(rs), 3) if rs[i:i+3].startswith("G")}
        xs.append(y); ys.append(len(seen))

    fig, ax = plt.subplots(figsize=DATASET_SQUARE_FIGSIZE)
    axb = ax.twinx()
    axb.hist(yr, bins=np.arange(1994, int(yr.max()) + 3, 1), color=VOCAB_BAR_C,
             edgecolor="white", lw=0.5, zorder=1)
    axb.set_ylabel("Structures deposited per year", color=VOCAB_BAR_AXIS_C)
    axb.tick_params(axis="y", colors=VOCAB_BAR_AXIS_C)
    axb.set_ylim(0, 15)                 # keep the bars reading as background
    axb.spines["top"].set_visible(False)

    ax.set_zorder(axb.get_zorder() + 1); ax.patch.set_visible(False)
    ax.step(xs, ys, where="post", color=VOCAB_C, lw=2.4, zorder=4)
    ax.scatter(xs, ys, s=12, color=VOCAB_C, alpha=0.5, lw=0, zorder=5)
    for lab, cut in sorted(MODEL_CUTOFFS.items(), key=lambda kv: kv[1]):
        c = pd.Timestamp(cut)
        x = c.year + (c.dayofyear - 1) / 365.25
        ax.axvline(x, color="#555555", ls="--", lw=1.1, zorder=3)
        left = lab in CUTOFF_LABEL_LEFT
        ax.annotate(f"{lab}  {cut}", (x, 2), xytext=(-11 if left else 11, 0),
                    textcoords="offset points", rotation=90, rotation_mode="anchor",
                    fontsize=8, color="#333333", fontweight="bold", ha="left",
                    va="top" if left else "bottom", zorder=6)
    ax.set_xlim(1996, yr.max() + 0.6)
    ax.set_ylabel("Cumulative distinct Gly-X-Y triplets", color=VOCAB_C)
    ax.tick_params(axis="y", colors=VOCAB_C)
    ax.set_xlabel("Deposition year")
    ax.set_title("Triplet vocabulary vs deposition rate")
    ax.grid(True, zorder=0)
    fig.tight_layout()
    save(fig, "Figure6_triplet_vocabulary")
    ctx.__exit__(None, None, None)


def dataset_figures():
    man = pd.read_csv(MANIFEST)
    print(f"Dataset figures over {len(man)} structures")
    fig_dataset_chain_length(man)
    fig_dataset_gxy_composition(man)
    fig_dataset_top_triplets(man)
    fig_dataset_identity_heatmap(man)
    fig_dataset_deposition_year(man)
    fig_dataset_triplet_vocabulary(man)


# ── fig 1: distributions ─────────────────────────────────────────────────────
def fig_distributions(piv, pdbs):
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6))
    axes = axes.ravel()
    for ax, (mcol, pretty, arrow, hib) in zip(axes, METRICS):
        data = [col(piv, v, mcol, pdbs) for v in ORDER]
        bp = ax.boxplot(data, positions=range(len(ORDER)), widths=0.6,
                        patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color="#888888", lw=1),
                        capprops=dict(color="#888888", lw=1),
                        boxprops=dict(lw=0))
        for patch, v in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLOR[v]); patch.set_alpha(0.28)
        for i, (v, d) in enumerate(zip(ORDER, data)):
            x = np.random.default_rng(i).normal(i, 0.055, len(d))
            ax.scatter(x, d, s=7, color=COLOR[v], alpha=0.7, lw=0, zorder=3)
            ax.scatter(i, np.mean(d), marker="D", s=34, color=COLOR[v],
                       edgecolor="white", lw=0.8, zorder=4)
        ax.set_title(f"{pretty}  {arrow}")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[v] for v in ORDER], rotation=32, ha="right")
        ax.grid(axis="y", zorder=0)
        if not hib:  # RMSD: clip extreme tail for readability, note it
            hi = np.nanpercentile(np.concatenate(data), 97)
            ax.set_ylim(0, hi * 1.15)
            ax.text(0.98, 0.96, "tail clipped at 97th pct", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color="#888888")
    axes[-1].axis("off")
    axes[-1].scatter([], [], marker="D", color="#666", label="mean")
    axes[-1].plot([], [], color=INK, lw=1.4, label="median")
    axes[-1].legend(loc="center", title="markers", title_fontsize=8)
    fig.suptitle("Score distributions across methods (75 shared targets)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig1_distributions")


# ── fig 2: side-chain gap ────────────────────────────────────────────────────
def fig_sidechain_gap(piv, pdbs):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bb_c, aa_c = "#4B4B4B", "#D55E00"
    ys = range(len(ORDER))
    for y, v in zip(ys, ORDER):
        bb = np.mean(col(piv, v, "global_lddt_backbone", pdbs))
        aa = np.mean(col(piv, v, "global_lddt_allatom", pdbs))
        ax.plot([aa, bb], [y, y], color="#BBBBBB", lw=2.2, zorder=1)
        ax.scatter(bb, y, s=70, color=bb_c, zorder=3, edgecolor="white", lw=0.8)
        ax.scatter(aa, y, s=70, color=aa_c, zorder=3, edgecolor="white", lw=0.8)
        ax.text(aa - 0.002, y + 0.22, f"Δ={bb-aa:.3f}", ha="right", va="bottom",
                fontsize=7.5, color=INK)
    ax.set_yticks(list(ys)); ax.set_yticklabels([LABEL[v] for v in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("mean lDDT")
    ax.set_title("Side-chain deficit: backbone vs all-atom lDDT")
    ax.scatter([], [], s=70, color=bb_c, label="backbone lDDT")
    ax.scatter([], [], s=70, color=aa_c, label="all-atom lDDT")
    ax.legend(loc="lower left")
    ax.grid(axis="x")
    ax.margins(x=0.12)
    fig.tight_layout()
    save(fig, "fig2_sidechain_gap")


# ── fig 2b: the same gap in RMSD ─────────────────────────────────────────────
def fig_sidechain_gap_rmsd(piv, pdbs):
    """RMSD counterpart of fig2. Medians, not means: per-target RMSD is strongly
    right-skewed (a few mis-registered structures dominate the mean), so medians
    describe the typical side-chain penalty."""
    with mpl.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        bb_c, aa_c = "#4B4B4B", "#D55E00"
        ys = range(len(ORDER))
        for y, v in zip(ys, ORDER):
            bb = np.median(col(piv, v, "global_rmsd_backbone", pdbs))
            aa = np.median(col(piv, v, "global_rmsd_allatom", pdbs))
            ax.plot([bb, aa], [y, y], color="#BBBBBB", lw=1.6, zorder=1)
            ax.scatter(bb, y, s=20, color=bb_c, zorder=3, edgecolor="white", lw=0.5)
            ax.scatter(aa, y, s=20, color=aa_c, zorder=3, edgecolor="white", lw=0.5)
            ax.text(aa + 0.025, y, f"Δ{aa-bb:+.2f}", ha="left", va="center",
                    fontsize=4.6, color=INK)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([SHORT[v] for v in ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("median RMSD (Å)  ↓")
        ax.set_title("Side-chain penalty")
        ax.scatter([], [], s=20, color=bb_c, label="backbone")
        ax.scatter([], [], s=20, color=aa_c, label="all-atom")
        ax.legend(loc="lower right", handletextpad=0.3, borderpad=0.3)
        ax.grid(axis="x")
        ax.margins(x=0.24, y=0.14)   # right margin leaves room for the Δ labels
        fig.subplots_adjust(**PAPER_ADJUST)
        save(fig, "fig2b_sidechain_gap_rmsd")


# ── fig 3: head-to-head scatter ──────────────────────────────────────────────
def fig_headtohead(piv, pdbs, kind):
    pairs = [("global_lddt_allatom", "lDDT (all-atom)", True),
             ("global_rmsd_backbone", "RMSD backbone (Å)", False)]
    opps = ["af3_nomsa", "boltz"]
    kcol = {"homotrimer": "#4477AA", "heterotrimer": "#EE6677"}
    kmk = {"homotrimer": "o", "heterotrimer": "^"}
    fig, axes = plt.subplots(2, len(opps), figsize=(4.0 * len(opps), 8))
    kinds = np.array([kind[p] for p in pdbs])
    # fixed display limits per metric so the RMSD outlier (~30 Å, on-diagonal) does
    # not crush the informative cluster; off-scale points are counted, not dropped.
    rvals = np.concatenate([col(piv, DET, "global_rmsd_backbone", pdbs)] +
                           [col(piv, o, "global_rmsd_backbone", pdbs) for o in opps])
    rcap = float(np.ceil(np.percentile(rvals, 95)))
    LIM = {"global_lddt_allatom": (0.45, 1.02), "global_rmsd_backbone": (0.0, rcap)}
    for r, (mcol, pretty, hib) in enumerate(pairs):
        x = col(piv, DET, mcol, pdbs)
        lo, hi = LIM[mcol]
        for c, opp in enumerate(opps):
            ax = axes[r, c]
            y = col(piv, opp, mcol, pdbs)
            ax.plot([lo, hi], [lo, hi], color="#999999", ls="--", lw=1, zorder=1)
            for k in ("homotrimer", "heterotrimer"):
                m = kinds == k
                ax.scatter(x[m], y[m], s=26, marker=kmk[k], color=kcol[k],
                           alpha=0.8, lw=0.4, edgecolor="white", zorder=3,
                           label=k if (r == 0 and c == 0) else None)
            det_better = (x > y) if hib else (x < y)   # win count over ALL points
            n = int(det_better.sum())
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            if hib:   # below diagonal is determ-better -> lower-right
                pos = (0.97, 0.05); ha, va = "right", "bottom"
            else:     # above diagonal is determ-better -> upper-left
                pos = (0.03, 0.97); ha, va = "left", "top"
            ax.text(*pos, f"Deterministic better:\n{n}/{len(pdbs)}",
                    transform=ax.transAxes, ha=ha, va=va, fontsize=8,
                    fontweight="bold", color="#D55E00")
            if not hib:
                off = int(((x > hi) | (y > hi)).sum())
                if off:
                    ax.text(0.97, 0.88, f"{off} off-scale", transform=ax.transAxes,
                            ha="right", va="top", fontsize=6.5, color="#888888")
            ax.set_xlabel(f"Deterministic — {pretty}")
            ax.set_ylabel(f"{LABEL[opp]} — {pretty}")
            ax.grid(True, zorder=0)
    axes[0, 0].legend(loc="upper left", title="target")
    fig.suptitle("Per-target head-to-head: Deterministic vs ML  (points off the "
                 "diagonal = a win)", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig3_headtohead_scatter")


# ── fig 4: ECDF of backbone RMSD ─────────────────────────────────────────────
def fig_ecdf(piv, pdbs):
    with mpl.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        xmax = 0
        for v in ORDER:
            d = np.sort(col(piv, v, "global_rmsd_backbone", pdbs))
            y = np.arange(1, len(d) + 1) / len(d)
            ax.step(d, y, where="post", color=COLOR[v], lw=1.2, ls=LS[v],
                    label=SHORT[v], zorder=3)
            xmax = max(xmax, np.percentile(d, 96))
        ax.axhline(0.5, color="#CCCCCC", lw=0.6, zorder=1)
        # to the right of the curves, which are all near 1.0 by then
        ax.text(xmax * 0.72, 0.52, "median", transform=ax.get_yaxis_transform(),
                fontsize=4.6, color="#999999")
        ax.set_xlim(0, xmax * 1.05)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("backbone RMSD (Å)  ↓")
        ax.set_ylabel("cumulative fraction of targets")
        ax.set_title("RMSD distribution (ECDF)")
        ax.grid(True, zorder=0)
        ax.legend(loc="lower right", handlelength=1.6, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.3)
        fig.subplots_adjust(**PAPER_ADJUST)
        save(fig, "fig4_ecdf_rmsd_bb")



def fig_bench_combined(piv, pdbs):
    """TM-score, RMSD and lDDT side by side, sharing one legend.

    Panels keep the standalone BENCH_FIGSIZE proportions, so this is the three
    individual figures tiled rather than a re-scaled layout.
    """
    w, h = BENCH_FIGSIZE
    legend_h = 0.42                       # strip reserved under the panels
    with mpl.rc_context(BENCH_RC):
        fig, axes = plt.subplots(1, 3, figsize=(w * 3, h + legend_h * 0.62))
        _bench_box(axes[0], piv, pdbs, ["tm_score"], [None], [None],
                   widths=0.46, offsets=[0.0])
        axes[0].set_title("TM-score  ↑")
        _bench_box(axes[1], piv, pdbs,
                   ["global_rmsd_backbone", "global_rmsd_allatom"],
                   [None, "////"], ["backbone", "all-atom"],
                   widths=0.28, offsets=[-0.17, 0.17])
        axes[1].set_title("RMSD (Å)  ↓")
        axes[1].set_ylim(top=12.5)          # matches the standalone fig_bench_rmsd
        axes[1].set_yticks(np.arange(0, 13, 2))
        _bench_box(axes[2], piv, pdbs,
                   ["global_lddt_backbone", "global_lddt_allatom"],
                   [None, "////"], ["backbone", "all-atom"],
                   widths=0.28, offsets=[-0.17, 0.17])
        axes[2].set_title("lDDT  ↑")
        handles = [mpl.patches.Patch(facecolor="#DDDDDD", edgecolor="#666666",
                                     lw=0.7, hatch=hh, label=ll)
                   for hh, ll in ((None, "backbone"), ("////", "all-atom"))]
        handles += [mpl.lines.Line2D([], [], color=INK, lw=1.0, label="median"),
                    mpl.lines.Line2D([], [], marker="D", ls="none", color="white",
                                     markeredgecolor=INK, markersize=3.6,
                                     label="mean")]
        # caption tucked under each title, without displacing the title itself
        for ax in axes:
            ax.text(0.5, 0.999, "75 shared targets", transform=ax.transAxes,
                    ha="center", va="top", fontsize=5.2, color="#888888")
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
                   bbox_to_anchor=(0.5, 0.030))
        fig.tight_layout(rect=[0, legend_h / (h + legend_h) * 0.62, 1, 1])
        save(fig, "fig_bench_combined")


# ═══ RMSD failure tail ═══════════════════════════════════════════════════════
# The ECDF (fig4) is clipped to the 96th percentile and shows the typical case.
# These three show the opposite end: how heavy each method's failure tail is, and
# whether the failures are the same targets.

TAIL_THRESH = (4.0, 6.0, 8.0, 10.0)
TAIL_METRIC = "global_rmsd_backbone"
# severity bands for the matrix: <4 (not a failure), 4-6, 6-8, 8-10, >10 A
TAIL_BANDS = [0, 4, 6, 8, 10, 1e9]
TAIL_COLORS = ["#F2F2F2", "#FDD9A8", "#F5A24B", "#DF6A2C", "#A32C0F"]


def fig_tail_survival(piv, pdbs):
    """P(RMSD > x) on a log axis — the tail an ECDF compresses into its corner."""
    n = len(pdbs)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for t in TAIL_THRESH:
        ax.axvline(t, color="#E0E0E0", lw=0.8, zorder=0)
    for v in ORDER:
        d = np.sort(col(piv, v, TAIL_METRIC, pdbs))
        surv = (n - np.arange(1, n + 1)) / n          # P(X > d_i)
        x = np.concatenate([[0.0], d])
        y = np.concatenate([[1.0], surv])
        keep = y > 0                                   # log axis: drop the final 0
        xk, yk = x[keep], y[keep]
        # S(x) is 1/n right up to the worst target; without this the curve would
        # stop at the *second* largest value and hide each method's worst failure.
        xk = np.append(xk, d[-1])
        yk = np.append(yk, yk[-1])
        ax.step(xk, yk, where="post", color=COLOR[v], lw=2, ls=LS[v],
                label=SHORT[v], zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.8 / n, 1.35)
    ticks = [1, 0.5, 0.2, 0.1, 0.05, 1 / n]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0%}" if t >= 0.05 else f"1/{n}" for t in ticks])
    ax.set_xlabel("backbone RMSD threshold x (Å)")
    ax.set_ylabel("fraction of targets with RMSD > x")
    ax.set_title("Failure tail: how often each method goes badly wrong")
    ax.text(0.99, 0.97, f"n = {n}; each curve ends at that method's worst target",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#888888")
    ax.grid(True, which="major", zorder=0)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "fig_tail_survival")


def _tail_failures(piv, pdbs, cutoff=4.0):
    """(targets, matrix) for every target where ANY method exceeds `cutoff`,
    ordered by worst-case RMSD descending."""
    vals = {v: piv[v].loc[pdbs, TAIL_METRIC] for v in ORDER}
    worst = pd.DataFrame(vals).max(axis=1)
    targets = list(worst[worst > cutoff].sort_values(ascending=False).index)
    mat = np.array([[vals[v][t] for t in targets] for v in ORDER])
    return targets, mat


def fig_tail_matrix(piv, pdbs):
    """Methods x failing targets. Shows that the failures are largely disjoint."""
    targets, mat = _tail_failures(piv, pdbs)
    cmap = mpl.colors.ListedColormap(TAIL_COLORS)
    norm = mpl.colors.BoundaryNorm(TAIL_BANDS, cmap.N)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.34 * len(targets)), 2.9))
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > TAIL_THRESH[0]:
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=5, color="#1A1A1A")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT[v] for v in ORDER], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    labels = ["< 4 Å", "4–6", "6–8", "8–10", "> 10 Å"]
    handles = [mpl.patches.Patch(facecolor=c, edgecolor="#CCCCCC", lw=0.5, label=l)
               for c, l in zip(TAIL_COLORS, labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.42),
              ncol=5, fontsize=7.5, title="backbone RMSD", title_fontsize=7.5)
    ax.set_title(f"Every target where any method exceeds 4 Å  "
                 f"({len(targets)} of {len(pdbs)})", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_tail_matrix")


def fig_tail_counts(piv, pdbs):
    """Grouped bars: one group per threshold, one column per method."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.16
    offs = (np.arange(len(ORDER)) - (len(ORDER) - 1) / 2) * w
    for k, v in enumerate(ORDER):
        d = col(piv, v, TAIL_METRIC, pdbs)
        counts = [int((d > t).sum()) for t in TAIL_THRESH]
        xs = np.arange(len(TAIL_THRESH)) + offs[k]
        ax.bar(xs, counts, width=w, color=COLOR[v], label=SHORT[v],
               edgecolor="white", lw=0.6, zorder=3)
        for x, c in zip(xs, counts):
            ax.text(x, c + 0.35, str(c), ha="center", va="bottom", fontsize=7,
                    color=INK if c else "#AAAAAA", zorder=4)
    ax.set_xticks(range(len(TAIL_THRESH)))
    ax.set_xticklabels([f"> {t:.0f} Å" for t in TAIL_THRESH])
    ax.set_xlabel("backbone RMSD threshold")
    ax.set_ylabel(f"targets exceeding threshold (of {len(pdbs)})")
    ax.set_title("Catastrophic failures by severity")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "fig_tail_counts")


# ═══ Accuracy by Gly-X-Y position ════════════════════════════════════════════
# score.py keys per-residue rows on the *reference* (pdb, chain, resnum) but does
# not record residue identity, so the Gly-X-Y position is recovered by re-reading
# the experimental CIFs (joins 100%).

THREE2ONE_REF = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "HYP": "O",
}


def gxy_map():
    """DataFrame(pdb_id, chain, resnum, gxy) for every reference residue. The frame
    is chosen per chain as the one with the most Gly at position 0 — the same rule
    step1_backbone_builder.reframe uses."""
    import gemmi
    exp = experimental_cif_dir()
    rows = []
    for fn in sorted(os.listdir(exp)):
        if not fn.endswith(".cif"):
            continue
        pid = fn[:-4].upper()
        st = gemmi.read_structure(os.path.join(exp, fn))
        st.setup_entities()
        for ch in st[0]:
            res = [r for r in ch if r.name in THREE2ONE_REF]
            if not res:
                continue
            seq = "".join(THREE2ONE_REF[r.name] for r in res)
            f = min(range(3),
                    key=lambda k: sum(1 for i in range(k, len(seq), 3) if seq[i] != "G"))
            for i, r in enumerate(res):
                rows.append((pid, ch.name, r.seqid.num, "GXY"[(i - f) % 3]))
    return (pd.DataFrame(rows, columns=["pdb_id", "chain", "resnum", "gxy"])
            .drop_duplicates(["pdb_id", "chain", "resnum"]))


def _gxy_paired_boot(sub, metric, order3=("G", "X", "Y"), n_boot=2000, seed=0):
    """Mean lDDT per triplet position expressed as a difference from Gly, with a
    95% CI bootstrapped over *structures*.

    Paired on purpose: all three positions occur in every chain, so the between-
    structure variance is common to them and cancels in the difference. CIs on the
    raw levels are ~10x wider and overlap for every method, which would wrongly
    suggest no positional effect. Absolute Gly levels are returned separately.
    """
    sub = sub.dropna(subset=[metric])
    pdbs = sorted(sub.pdb_id.unique())
    # per structure, per position: the residue-level values
    cells = {p: [sub[(sub.pdb_id == q) & (sub.gxy == p)][metric].to_numpy()
                 for q in pdbs] for p in order3}

    def pooled(pos, take):
        vals = np.concatenate([cells[pos][i] for i in take])
        return vals.mean() if len(vals) else np.nan

    full = np.arange(len(pdbs))
    gly = pooled("G", full)
    point = {p: pooled(p, full) - gly for p in order3}
    rng = np.random.default_rng(seed)
    draws = {p: [] for p in order3}
    for _ in range(n_boot):
        take = rng.choice(full, len(full))
        g = pooled("G", take)
        for p in order3:
            draws[p].append(pooled(p, take) - g)
    ci = {p: (np.nanpercentile(draws[p], 2.5), np.nanpercentile(draws[p], 97.5))
          for p in order3}
    return gly, point, ci


def fig_gxy_position(pdbs):
    pr = pd.read_csv(os.path.join(RES, "scores_per_residue.csv"))
    pr = pr[pr.variant.isin(ORDER) & pr.pdb_id.isin(pdbs)]
    pr = pr.merge(gxy_map(), on=["pdb_id", "chain", "resnum"])
    order3 = ["G", "X", "Y"]
    panels = [("lddt_allatom", "all-atom lDDT"), ("lddt_backbone", "backbone lDDT")]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, (metric, pretty) in zip(axes, panels):
        for v in ORDER:
            gly, point, ci = _gxy_paired_boot(pr[pr.variant == v], metric,
                                              tuple(order3))
            m = [point[p] for p in order3]
            lo = [m[i] - ci[p][0] for i, p in enumerate(order3)]
            hi = [ci[p][1] - m[i] for i, p in enumerate(order3)]
            ax.errorbar(range(3), m, yerr=[lo, hi], color=COLOR[v], lw=2, ls=LS[v],
                        marker="o", ms=5, capsize=3, elinewidth=1,
                        label=f"{SHORT[v]}  (Gly {gly:.3f})", zorder=3)
        ax.axhline(0, color="#999999", lw=0.9, zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Gly", "X", "Y"])
        ax.set_xlim(-0.35, 2.35)
        ax.set_xlabel("position in the Gly-X-Y triplet")
        ax.set_title(pretty)
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("change in mean per-residue lDDT vs Gly\n"
                       "(95% CI, paired bootstrap over targets)")
    axes[0].legend(loc="lower left", ncol=1)
    n = pr[pr.variant == DET].gxy.value_counts()
    fig.suptitle("Accuracy by position in the triplet — the positional penalty is "
                 "entirely a side-chain effect\n"
                 f"(n = {n.get('G', 0)} Gly, {n.get('X', 0)} X, {n.get('Y', 0)} Y "
                 "residues per method; absolute Gly lDDT in the legend)",
                 fontweight="bold", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_gxy_position")


# ── fig 6: positional lDDT profiles ──────────────────────────────────────────
def _positional_bins(metric, pdbs, nb=20):
    """Per-residue table binned by fractional position along each chain (N->C)."""
    pr = pd.read_csv(os.path.join(RES, "scores_per_residue.csv"))
    pr = pr[pr.variant.isin(ORDER) & pr.pdb_id.isin(pdbs)].dropna(subset=[metric])
    g = pr.groupby(["pdb_id", "variant", "chain"])["resnum"]
    lo = g.transform("min"); hi = g.transform("max")
    span = (hi - lo).replace(0, np.nan)
    pr = pr.assign(pos=(pr.resnum - lo) / span).dropna(subset=["pos"])
    edges = np.linspace(0, 1, nb + 1)
    pr["bin"] = np.clip(np.digitize(pr.pos, edges) - 1, 0, nb - 1)
    return pr, (edges[:-1] + edges[1:]) / 2, nb


def _fig_positional_lddt(pdbs, metric, ylabel, name, ylim=None):
    """Mean +/- SEM lDDT profile. lDDT is bounded and near-symmetric per bin, so
    unlike the RMSD profile (fig6b) the mean and its SEM band are well behaved."""
    pr, centers, nb = _positional_bins(metric, pdbs)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for v in ORDER:
        sub = pr[pr.variant == v].groupby("bin")[metric]
        mean = sub.mean().reindex(range(nb))
        sem = sub.sem().reindex(range(nb))
        ax.plot(centers, mean, color=COLOR[v], lw=2, ls=LS[v], label=LABEL[v], zorder=3)
        ax.fill_between(centers, mean - sem, mean + sem, color=COLOR[v], alpha=0.12, lw=0)
    ax.set_xlabel("fractional position along chain (N → C)")
    ax.set_ylabel(ylabel)
    ax.set_title("Where accuracy is lost along the helix (terminal fraying)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, zorder=0)
    ax.legend(loc="lower center", ncol=3)
    ax.margins(x=0.02)
    fig.tight_layout()
    save(fig, name)


def fig_positional(pdbs):
    _fig_positional_lddt(pdbs, "lddt_allatom",
                         "mean per-residue lDDT (all-atom)",
                         "fig6_positional_lddt")


def fig_positional_lddt_bb(pdbs):
    _fig_positional_lddt(pdbs, "lddt_backbone",
                         "mean per-residue lDDT (backbone)",
                         "fig6c_positional_lddt_bb", ylim=(0.88, 1.00))


# ── fig 6b: positional backbone RMSD ─────────────────────────────────────────
def fig_positional_rmsd(pdbs):
    """Backbone-RMSD counterpart of fig6. Per-residue RMSD is strongly right-skewed
    (median 0.73 Å, mean 1.89 Å, max 24.5 Å), so a mean±SEM profile would track a
    handful of mis-registered structures rather than the terminal-fraying signal.
    Median with an interquartile band instead."""
    pr = pd.read_csv(os.path.join(RES, "scores_per_residue.csv"))
    pr = pr[pr.variant.isin(ORDER) & pr.pdb_id.isin(pdbs)].dropna(subset=["rmsd_backbone"])
    g = pr.groupby(["pdb_id", "variant", "chain"])["resnum"]
    lo = g.transform("min"); hi = g.transform("max")
    span = (hi - lo).replace(0, np.nan)
    pr = pr.assign(pos=(pr.resnum - lo) / span).dropna(subset=["pos"])
    nb = 20
    edges = np.linspace(0, 1, nb + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    pr["bin"] = np.clip(np.digitize(pr.pos, edges) - 1, 0, nb - 1)
    with mpl.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        # Median only: per-bin quartiles are dominated by which structures land in
        # each bin, and an IQR band swamps the profile without adding information.
        for v in ORDER:
            med = (pr[pr.variant == v].groupby("bin")["rmsd_backbone"]
                   .median().reindex(range(nb)))
            ax.plot(centers, med, color=COLOR[v], lw=1.2, ls=LS[v],
                    label=SHORT[v], zorder=3)
        ax.set_xlabel("fractional position (N → C)")
        ax.set_ylabel("median backbone RMSD (Å)  ↓")
        ax.set_title("Terminal fraying")
        ax.grid(True, zorder=0)
        ax.legend(loc="upper center", ncol=2, handlelength=1.6, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.3, columnspacing=1.0)
        ax.margins(x=0.02)
        fig.subplots_adjust(**PAPER_ADJUST)
        save(fig, "fig6b_positional_rmsd_bb")


# ── win-rate stacked bars ────────────────────────────────────────────────────
def fig_winrate_bars(piv, pdbs):
    # TM pinned top (requested), then strong -> weak for deterministic
    rows = [("tm_score", "TM-score", True),
            ("global_rmsd_backbone", "RMSD backbone", False),
            ("global_rmsd_allatom", "RMSD all-atom", False),
            ("global_lddt_backbone", "lDDT backbone", True),
            ("global_lddt_allatom", "lDDT all-atom", True)]
    methods = list(ML)                # fixed order within each group
    det_c = COLOR[DET]
    fig, ax = plt.subplots(figsize=(8.5, 1.55 * len(ML) * len(rows) / 4 + 1.9))
    y = 0.0
    group_mid = []
    for mcol, mlabel, hib in rows:
        g0 = y
        d = col(piv, DET, mcol, pdbs)
        for opp in methods:
            o = col(piv, opp, mcol, pdbs)
            det_better = (d > o) if hib else (d < o)
            ml_better = (o > d) if hib else (o < d)
            nz = int(det_better.sum() + ml_better.sum())
            wr = det_better.sum() / nz if nz else 0.5     # deterministic win fraction
            # left = deterministic share (vermillion); right = this model's share (its colour)
            ax.barh(y, wr, color=det_c, edgecolor="white", lw=1.4, height=0.80, zorder=3)
            ax.barh(y, 1 - wr, left=wr, color=_light(COLOR[opp]),
                    edgecolor="white", lw=1.4, height=0.80, zorder=3)
            if wr >= 0.14:                                 # % inside the vermillion segment
                ax.text(wr / 2, y, f"{wr*100:.0f}%", va="center", ha="center",
                        color="white", fontsize=8.5, fontweight="bold", zorder=5)
            else:                                          # too narrow -> place just outside
                ax.text(wr + 0.012, y, f"{wr*100:.0f}%", va="center", ha="left",
                        color=INK, fontsize=8.5, fontweight="bold", zorder=5)
            ax.text(1.015, y, LABEL[opp], va="center", ha="left", fontsize=8.5,
                    transform=ax.get_yaxis_transform())
            y += 1
        group_mid.append((mlabel, (g0 + y - 1) / 2))
        y += 0.9
    ax.axvline(0.5, color="#555555", lw=1.1, ls="--", zorder=4)
    for mlabel, mid in group_mid:
        ax.text(-0.02, mid, mlabel, va="center", ha="right", fontsize=11.5,
                fontweight="bold", transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("deterministic win rate (%)  —  vs each ML model, per target (ties excluded)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=det_c, label="Deterministic wins"),
               mpatches.Patch(color="#C9C9C9", label="ML model wins (right segment, model's colour)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=2, frameon=False, fontsize=9)
    ax.set_title("Per-target win rate: Deterministic vs each ML model", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_winrate_bars")


# ── paired win-margin (RMSD backbone) ────────────────────────────────────────
def fig_winmargin(piv, pdbs):
    fig, axes = plt.subplots(1, len(ML), figsize=(3.0 * len(ML), 3.8), sharey=True)
    for ax, opp in zip(axes, ML):
        d = col(piv, DET, "global_rmsd_backbone", pdbs)
        o = col(piv, opp, "global_rmsd_backbone", pdbs)
        diff = d - o                       # <0  -> deterministic better (lower RMSD)
        order = np.argsort(diff)
        diff = diff[order]
        colors = np.where(diff < 0, "#D55E00", "#9AA0A6")
        ax.bar(range(len(diff)), diff, color=colors, width=1.0, lw=0)
        ax.axhline(0, color=INK, lw=0.8)
        nwin = int((diff < 0).sum())
        med = np.median(diff)
        try:
            p = wilcoxon(d, o).pvalue
            ptxt = f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"
        except ValueError:
            ptxt = "p=n/a"
        ax.set_title(f"vs {LABEL[opp]}", fontsize=10)
        ax.text(0.03, 0.04,
                f"Determ wins {nwin}/{len(diff)}\nmedian Δ={med:+.2f} Å\nWilcoxon {ptxt}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5,
                color=INK)
        ax.set_xlabel("targets (sorted)")
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("ΔRMSD backbone (Å)\nDeterministic − ML")
    # symmetric-ish y limits, clip extreme
    allv = np.concatenate([col(piv, DET, "global_rmsd_backbone", pdbs) -
                           col(piv, o, "global_rmsd_backbone", pdbs) for o in ML])
    lim = np.percentile(np.abs(allv), 96)
    for ax in axes:
        ax.set_ylim(-lim, lim)
    fig.suptitle("Paired per-target backbone-RMSD margin  (orange = deterministic wins; "
                 "below 0 = deterministic lower RMSD)", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig_winmargin_rmsd_bb")


# ═══ CDSM vs ML benchmark, stage-figure style ════════════════════════════════
# Same visual grammar as the CDSM stage figures (Tukey boxes, all points jittered,
# median = box line, mean = white diamond), but one group per method and pastel
# per-method colours. Sized so all three tile across one page width (3 x 2.8 in).

BENCH_FIGSIZE = (2.8, 3.0)
SHORT = {DET: "CDSM", "boltz": "Boltz-2", "chai": "Chai-1",
         "protenix": "Protenix", "af3_nomsa": "AF3"}
BENCH_RC = {"font.size": 6.5, "axes.titlesize": 7.5, "xtick.labelsize": 5.5,
            "ytick.labelsize": 6.5, "axes.labelsize": 6, "legend.fontsize": 5.5}


def _bench_box(ax, piv, pdbs, keys, hatches, labels, widths, offsets):
    pos = np.arange(len(ORDER), dtype=float)
    for k, (key, hatch, off) in enumerate(zip(keys, hatches, offsets)):
        data = [col(piv, v, key, pdbs) for v in ORDER]
        for i, (v, d) in enumerate(zip(ORDER, data)):
            p = pos[i] + off
            ax.boxplot([d], positions=[p], widths=widths, whis=1.5,
                       showfliers=False, patch_artist=True, zorder=2,
                       medianprops=dict(color=INK, lw=1.0),
                       boxprops=dict(facecolor=_light(COLOR[v], 0.5), alpha=0.9,
                                     edgecolor=COLOR[v], lw=0.7, hatch=hatch),
                       whiskerprops=dict(color=COLOR[v], lw=0.7),
                       capprops=dict(color=COLOR[v], lw=0.7))
            x = np.random.default_rng(100 * k + i).normal(p, widths * 0.16, len(d))
            ax.scatter(x, d, s=2.2, color=COLOR[v], alpha=0.55, lw=0, zorder=3)
            ax.scatter(p, np.mean(d), marker="D", s=11, color="white",
                       edgecolor=INK, lw=0.6, zorder=5)
    ax.set_xticks(pos)
    # wrap the long ablation label so the horizontal ticks do not collide
    ax.set_xticklabels([SHORT[v].replace(" no-MSA", "\nno-MSA") for v in ORDER],
                       rotation=0, ha="center")
    ax.set_xlim(pos[0] - 0.62, pos[-1] + 0.62)
    ax.grid(axis="y", zorder=0)


def fig_bench_tmscore(piv, pdbs):
    with mpl.rc_context(BENCH_RC):
        fig, ax = plt.subplots(figsize=BENCH_FIGSIZE)
        _bench_box(ax, piv, pdbs, ["tm_score"], [None], [None],
                   widths=0.46, offsets=[0.0])
        ax.set_title("TM-score  ↑")
        fig.tight_layout()
        save(fig, "fig_bench_tmscore")


def fig_bench_lddt(piv, pdbs):
    with mpl.rc_context(BENCH_RC):
        fig, ax = plt.subplots(figsize=BENCH_FIGSIZE)
        _bench_box(ax, piv, pdbs,
                   ["global_lddt_backbone", "global_lddt_allatom"],
                   [None, "////"], ["backbone", "all-atom"],
                   widths=0.28, offsets=[-0.17, 0.17])
        ax.set_title("lDDT  ↑")
        fig.tight_layout()
        save(fig, "fig_bench_lddt")


def fig_bench_rmsd(piv, pdbs):
    with mpl.rc_context(BENCH_RC):
        fig, ax = plt.subplots(figsize=BENCH_FIGSIZE)
        _bench_box(ax, piv, pdbs,
                   ["global_rmsd_backbone", "global_rmsd_allatom"],
                   [None, "////"], ["backbone", "all-atom"],
                   widths=0.28, offsets=[-0.17, 0.17])
        ax.set_title("RMSD (Å)  ↓")
        ax.set_ylim(top=12.5)
        ax.set_yticks(np.arange(0, 13, 2))
        fig.tight_layout()
        save(fig, "fig_bench_rmsd")



# ═══ Training-cutoff analysis ════════════════════════════════════════════════
# Each ML model has a training-data cutoff; CDSM, being deterministic, has none
# and therefore acts as the control. If a model's advantage shrinks on structures
# deposited after its cutoff while CDSM's scores hold steady, the shift cannot be
# explained by the post-cutoff structures simply being harder.

CUTOFF = {"af3_nomsa": "2021-09-30", "protenix": "2021-09-30",
          "chai": "2021-01-12", "boltz": "2023-06-01"}
CUTOFF_ORDER = ["af3_nomsa", "protenix", "chai", "boltz"]
# Boltz-2 needs the full RMSD range; the others read better zoomed in.
CUTOFF_RMSD_YLIM = {"af3_nomsa": (0.7, 1.6), "protenix": (0.7, 1.6),
                    "chai": (0.7, 1.6), "boltz": (0.0, 6.0)}
CUTOFF_RC = {"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
             "xtick.labelsize": 7, "ytick.labelsize": 7}
# (row label, metric column, higher_is_better, metric ylim, win-rate ylim)
CUTOFF_ROWS = [
    ("TM-score", "tm_score", True, (0.85, 1.0), (20, 80)),
    ("RMSD (Å)", "global_rmsd_backbone", False, (0.0, 6.0), (20, 80)),
    ("lDDT", "global_lddt_backbone", True, (0.970, 1.0), (0, 30)),
]


def _cutoff_era(pdbs, dates, variant, which):
    cut = pd.Timestamp(CUTOFF[variant])
    return [p for p in pdbs
            if (dates[p] <= cut if which == "pre" else dates[p] > cut)]


def _cutoff_winrate(piv, ids, variant, key, hib):
    d = piv[DET].loc[ids, key].to_numpy()
    o = piv[variant].loc[ids, key].to_numpy()
    w = int((d > o).sum() if hib else (d < o).sum())
    l = int((o > d).sum() if hib else (o < d).sum())
    return 100.0 * w / (w + l) if w + l else np.nan


def _declash(items, span, top):
    """Nudge labels apart so their text does not overlap, keeping them in order."""
    gap = 0.062 * span
    placed = []
    for val, colour in sorted(items, key=lambda it: it[0]):
        y = val if not placed else max(val, placed[-1][1] + gap)
        placed.append((val, y, colour))
    over = placed[-1][1] - top if placed and placed[-1][1] > top else 0.0
    return [(v, y - over, c) for v, y, c in placed]


def fig_cutoff_grid(piv, pdbs):
    """3 metric rows x (CDSM win rate + one panel per model), pre vs post cutoff."""
    man = pd.read_csv(MANIFEST)
    man["pdb_id"] = man.pdb_id.str.upper()
    dates = dict(zip(man.pdb_id, pd.to_datetime(man.deposition_date)))

    ctx = mpl.rc_context(CUTOFF_RC); ctx.__enter__()
    fig, axes = plt.subplots(3, 6, figsize=(11.2, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 0.06, 1, 1, 1, 1],
                                          "hspace": 0.28, "wspace": 0.32})
    for r in range(3):
        axes[r, 1].axis("off")        # spacer between win rate and the models

    for r, (rlab, key, hib, ylim, wlim) in enumerate(CUTOFF_ROWS):
        ax = axes[r, 0]
        labels = {"pre": [], "post": []}
        for v in CUTOFF_ORDER:
            y = [_cutoff_winrate(piv, _cutoff_era(pdbs, dates, v, e), v, key, hib)
                 for e in ("pre", "post")]
            ax.plot([0, 1], y, color=COLOR[v], lw=0.8, marker="o", ms=3.2, zorder=3)
            labels["pre"].append((y[0], COLOR[v]))
            labels["post"].append((y[1], COLOR[v]))
        span = wlim[1] - wlim[0]
        for side, x, ha, bold in (("pre", -0.12, "right", False),
                                  ("post", 1.12, "left", True)):
            for val, y, colour in _declash(labels[side], span, wlim[1]):
                ax.text(x, y, f"{val:.0f}%", ha=ha, va="center", fontsize=6.5,
                        color=colour, fontweight="bold" if bold else "normal")
        ax.axhline(50, color="#999999", ls="--", lw=0.8, zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
        ax.set_xlim(-0.45, 1.45); ax.set_ylim(*wlim)
        ax.set_ylabel("CDSM win rate (%)")
        ax.grid(axis="y", zorder=0, lw=0.5)
        if r == 0:
            ax.set_title("CDSM win rate", fontweight="bold")

        for j, v in enumerate(CUTOFF_ORDER):
            ax = axes[r, j + 2]
            pre = _cutoff_era(pdbs, dates, v, "pre")
            post = _cutoff_era(pdbs, dates, v, "post")
            plim = CUTOFF_RMSD_YLIM[v] if key.startswith("global_rmsd") else ylim
            for who in (v, DET):
                y = [min(piv[who].loc[ids, key].median(), plim[1])
                     for ids in (pre, post)]
                ax.plot([0, 1], y, color=COLOR[who], lw=0.8, marker="o", ms=3.2,
                        zorder=3)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
            ax.set_xlim(-0.25, 1.25); ax.set_ylim(*plim)
            ax.grid(axis="y", zorder=0, lw=0.5)
            if j == 0:
                ax.set_ylabel(rlab)
            if r == 0:
                ax.set_title(SHORT[v], fontsize=8, fontweight="bold", pad=12)
                ax.text(0.5, 1.03, f"{CUTOFF[v]}   n={len(pre)}/{len(post)}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.8, color="#555555")

    handles = [mpl.lines.Line2D([], [], color=COLOR[v], lw=1.8, label=SHORT[v])
               for v in CUTOFF_ORDER]
    handles.append(mpl.lines.Line2D([], [], color=COLOR[DET], lw=1.8, label="CDSM"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("CDSM vs ML across each model's training cutoff  "
                 "(backbone metrics, medians)", fontweight="bold", y=1.0,
                 fontsize=9.5)
    fig.tight_layout(rect=[0.022, 0.055, 1, 0.98])
    for r, (rlab, *_rest) in enumerate(CUTOFF_ROWS):
        box = axes[r, 0].get_position()
        fig.text(box.x0 - 0.052, box.y0 + box.height / 2, rlab, rotation=90,
                 va="center", ha="center", fontsize=9.5, fontweight="bold")
    save(fig, "fig_cutoff_grid")
    ctx.__exit__(None, None, None)


# ═══ CDSM pipeline-stage figures ═════════════════════════════════════════════
# These cover the deterministic builder alone (ML methods excluded): how the
# scores move as the build progresses core -> full sequence -> re-registered ->
# relaxed. Side chains are built by tleap at *every* stage (run_pipeline.py calls
# step4 for each output), so all-atom metrics are defined throughout; the last
# stage adds only the restrained OpenMM minimisation.


def _reframe_info(seq):
    """(reframed_seq, n_dropped, c_dropped) — mirrors step1_backbone_builder.reframe:
    shift to the frame with the most Gly at position 0, drop the trailing partial
    triplet. Inlined so this script stays self-contained."""
    s = str(seq).upper().strip().replace("﻿", "")
    f = min(range(3), key=lambda k: sum(1 for i in range(k, len(s), 3) if s[i] != "G"))
    shifted = s[f:]
    keep = len(shifted) - (len(shifted) % 3)
    return shifted[:keep], f, len(shifted) - keep


def native_ids():
    """PDB IDs THeBuScr builds *fully natively*: every chain already in Gly-X-Y
    register (nothing trimmed at either terminus) and all three reframed chains
    the same length — so no terminal extension and no unequal-length completion.
    Excludes the 5 structures the builder cannot handle at all."""
    man = pd.read_csv(MANIFEST)
    out = set()
    for _, r in man.iterrows():
        pid = str(r["pdb_id"]).strip().upper()
        if pid in KNOWN_ZERODIV:
            continue
        seqs = ([r["chain_a_sequence"]] * 3
                if str(r["kind"]).strip().lower() == "homotrimer"
                else [r["chain_a_sequence"], r["chain_b_sequence"], r["chain_c_sequence"]])
        info = [_reframe_info(s) for s in seqs]
        if all(nd == 0 and cd == 0 for _, nd, cd in info) and \
                len({len(a) for a, _, _ in info}) == 1:
            out.add(pid)
    return out


def load_cdsm():
    """[(label_with_n, DataFrame), ...] for the 5 pipeline stages."""
    s = pd.read_csv(os.path.join(RES, "scores_summary.csv"))
    nat = native_ids()
    groups = []
    for label, variant, only_native in CDSM_STAGES:
        d = s[s.variant == variant]
        if only_native:
            d = d[d.pdb_id.isin(nat)]
        groups.append((f"{label}\n(n={len(d)})", d))
    return groups


def _cdsm_box(ax, groups, keys, colors, labels, widths, offsets):
    """Tukey boxes (whis=1.5) with every datapoint jittered on top; the box line is
    the MEDIAN, the white diamond is the MEAN. One series per (key, offset)."""
    pos = np.arange(len(groups), dtype=float)
    for k, (key, color, off) in enumerate(zip(keys, colors, offsets)):
        data = [g[key].dropna().to_numpy() for _, g in groups]
        ax.boxplot(data, positions=pos + off, widths=widths, whis=1.5,
                   showfliers=False, patch_artist=True, zorder=2,
                   medianprops=dict(color=INK, lw=1.5),
                   boxprops=dict(facecolor=color, alpha=0.28, edgecolor=color, lw=1),
                   whiskerprops=dict(color=color, lw=1),
                   capprops=dict(color=color, lw=1))
        for i, (p, d) in enumerate(zip(pos + off, data)):
            x = np.random.default_rng(100 * k + i).normal(p, widths * 0.16, len(d))
            ax.scatter(x, d, s=9, color=color, alpha=0.55, lw=0.2,
                       edgecolor="white", zorder=3)
            ax.scatter(p, np.mean(d), marker="D", s=32, color="white",
                       edgecolor=INK, lw=0.9, zorder=5)
    ax.set_xticks(pos)
    ax.set_xticklabels([n for n, _ in groups])
    ax.set_xlim(pos[0] - 0.62, pos[-1] + 0.62)
    ax.grid(axis="y", zorder=0)
    handles = [mpl.patches.Patch(facecolor=c, alpha=0.35, edgecolor=c, label=l)
               for c, l in zip(colors, labels) if l]
    handles += [mpl.lines.Line2D([], [], color=INK, lw=1.5, label="median"),
                mpl.lines.Line2D([], [], marker="D", ls="none", color="white",
                                 markeredgecolor=INK, markersize=6, label="mean")]
    ax.legend(handles=handles, loc="best", ncol=2)


def fig_cdsm_tmscore(groups):
    fig, ax = plt.subplots(figsize=CDSM_FIGSIZE)
    _cdsm_box(ax, groups, ["tm_score"], [STAGE_C], [None],
              widths=0.42, offsets=[0.0])
    ax.set_ylabel("TM-score  ↑")
    ax.set_title("CDSM pipeline — TM-score by build stage")
    fig.tight_layout()
    save(fig, "fig_cdsm_tmscore")


def fig_cdsm_lddt(groups):
    fig, ax = plt.subplots(figsize=CDSM_FIGSIZE)
    _cdsm_box(ax, groups,
              ["global_lddt_backbone", "global_lddt_allatom"], [BB_C, AA_C],
              ["backbone", "all-atom"], widths=0.26, offsets=[-0.16, 0.16])
    ax.set_ylabel("lDDT  ↑")
    ax.set_title("CDSM pipeline — lDDT by build stage")
    fig.tight_layout()
    save(fig, "fig_cdsm_lddt")


def fig_cdsm_rmsd(groups):
    fig, ax = plt.subplots(figsize=CDSM_FIGSIZE)
    _cdsm_box(ax, groups,
              ["global_rmsd_backbone", "global_rmsd_allatom"], [BB_C, AA_C],
              ["backbone", "all-atom"], widths=0.26, offsets=[-0.16, 0.16])
    ax.set_ylabel("RMSD (Å)  ↓")          # autoscaled: no cap, no clipped tail
    ax.set_title("CDSM pipeline — RMSD by build stage")
    fig.tight_layout()
    save(fig, "fig_cdsm_rmsd")


def table_cdsm(groups):
    """Metrics as rows, stages as columns; cells = median. CSV + markdown."""
    cols = [f"{g.replace(chr(10), ' ')}" for g, _ in groups]
    df = pd.DataFrame(
        [[round(float(np.median(g[key].dropna().to_numpy())), 3) for _, g in groups]
         for key, _ in CDSM_METRICS],
        index=[l for _, l in CDSM_METRICS], columns=cols)
    df.index.name = "Metric (median)"
    out = os.path.join(RES, "cdsm_stage_table.csv")
    df.to_csv(out)
    print(f"  wrote {os.path.relpath(out, ROOT)}")

    print("\n| Metric (median) | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for label, row in df.iterrows():
        print(f"| {label} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    print()


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    dataset_figures()

    s, piv, pdbs, kind = load()
    print(f"Building figures over {len(pdbs)} shared targets → {HERE}")
    fig_distributions(piv, pdbs)
    fig_sidechain_gap(piv, pdbs)
    fig_sidechain_gap_rmsd(piv, pdbs)
    fig_headtohead(piv, pdbs, kind)
    fig_ecdf(piv, pdbs)
    fig_tail_survival(piv, pdbs)
    fig_tail_matrix(piv, pdbs)
    fig_tail_counts(piv, pdbs)
    fig_positional(pdbs)
    fig_positional_rmsd(pdbs)
    fig_positional_lddt_bb(pdbs)
    fig_gxy_position(pdbs)
    fig_winrate_bars(piv, pdbs)
    fig_winmargin(piv, pdbs)
    fig_bench_tmscore(piv, pdbs)
    fig_bench_lddt(piv, pdbs)
    fig_bench_rmsd(piv, pdbs)
    fig_bench_combined(piv, pdbs)
    fig_cutoff_grid(piv, pdbs)

    groups = load_cdsm()
    print("CDSM stage figures: " +
          ", ".join(g.replace("\n", " ") for g, _ in groups))
    fig_cdsm_tmscore(groups)
    fig_cdsm_lddt(groups)
    fig_cdsm_rmsd(groups)
    table_cdsm(groups)
    print("done.")


if __name__ == "__main__":
    main()
