"""
Network-sorted connectivity comparison for the ABIDE I diploma project.

CC200 (Craddock-200) is a *data-driven* parcellation and ships **without**
network labels. The standard way to display CC200 functional-connectivity
matrices in neuroscience papers is therefore to:

  1. Take an atlas that *does* carry Yeo 2011 7-network labels — here we
     use **Schaefer 2018, 400 parcels, 7-networks**, downloaded from the
     ThomasYeoLab/CBIG GitHub mirror.
  2. For each CC200 ROI, compute its centroid in MNI space and look up
     the Yeo network at that location (winner-takes-all if the centroid
     voxel itself is unlabelled).
  3. Permute the 200×200 connectivity matrix so ROIs of the same network
     sit in contiguous blocks, in the canonical Yeo order:
         Visual → Somatomotor → DorsAttn → VentAttn (Salience) →
         Limbic → Frontoparietal (Control) → Default Mode → Unassigned
  4. Re-plot mean-ASD, mean-Control and the difference matrix with
     **white grid lines** between the network blocks and a **network
     label** on each block.

Outputs (in ``visualizations/``):
  - connectivity_comparison_networks.png
  - connectivity_comparison_networks.html
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout so progress strings with arrows / Greek letters work
# under Windows' default cp1251 console codec.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nilearn import datasets, image
from plotly.subplots import make_subplots
from scipy import ndimage as ndi

RANDOM_STATE = 42
N_ROIS = 200

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
VIS_DIR = PROJECT_DIR / "visualizations"

DX_LABELS = {1: "ASD", 2: "Control"}

# Canonical display order + colors for the Yeo 7 networks (matches the
# colour scheme used in most network-connectivity publications).
NETWORK_ORDER = [
    "Visual", "Somatomotor", "DorsAttn", "VentAttn",
    "Limbic", "Frontoparietal", "Default", "Unassigned",
]
NETWORK_COLORS = {
    "Visual":         "#781286",
    "Somatomotor":    "#4682B4",
    "DorsAttn":       "#00760E",
    "VentAttn":       "#C43AFA",
    "Limbic":         "#DCF8A4",
    "Frontoparietal": "#E69422",
    "Default":        "#CD3E4E",
    "Unassigned":     "#7F7F7F",
}

# Schaefer label naming uses these abbreviations; map to our display names.
SCHAEFER_ABBREV = {
    "Vis":         "Visual",
    "SomMot":      "Somatomotor",
    "DorsAttn":    "DorsAttn",
    "SalVentAttn": "VentAttn",
    "Limbic":      "Limbic",
    "Cont":        "Frontoparietal",
    "Default":     "Default",
}


# --------------------------------------------------------------------------- #
# Atlas → network mapping                                                     #
# --------------------------------------------------------------------------- #


def build_cc200_to_network() -> List[str]:
    """Return a list of length ``N_ROIS`` mapping ROI index → network name."""
    print("[1/4] Building CC200 → Yeo-7 network mapping ...")

    from src.utils.atlas_fetch import robust_fetch

    # CC200 labelmap (idx 19 = ~200-cluster solution, see visualize_brain.py).
    # nitrc.org ships an invalid cert → fetch via the SSL-robust wrapper.
    craddock = robust_fetch(
        datasets.fetch_atlas_craddock_2012,
        data_dir=str(DATA_DIR),
        partial_dirs=[DATA_DIR / "craddock_2012"],
    )
    cc_img = image.index_img(craddock["maps"], 19)
    cc_data = cc_img.get_fdata().astype(int)

    # Schaefer 2018, 400 parcels, 7 networks, MNI152 2mm.
    sch = robust_fetch(
        datasets.fetch_atlas_schaefer_2018,
        n_rois=400, yeo_networks=7, resolution_mm=2,
        data_dir=str(DATA_DIR),
        partial_dirs=[DATA_DIR / "schaefer_2018"],
    )
    sch_img_native = image.load_img(sch["maps"])

    # Resample Schaefer onto the CC200 grid so we can look up network labels
    # at CC200 voxel coordinates directly.
    sch_resampled = image.resample_to_img(
        sch_img_native, cc_img, interpolation="nearest", copy_header=True,
    )
    sch_data = sch_resampled.get_fdata().astype(int)

    # Schaefer label -> network display name.
    # Index in `sch["labels"]` IS the integer label value (0 = Background).
    sch_labels = list(sch["labels"])
    label_to_net: Dict[int, str] = {}
    for i, raw in enumerate(sch_labels):
        raw = str(raw)
        if raw == "Background":
            continue
        # Format: "7Networks_LH_Vis_1" → abbrev = "Vis"
        parts = raw.split("_")
        abbrev = parts[2] if len(parts) >= 3 else ""
        label_to_net[i] = SCHAEFER_ABBREV.get(abbrev, "Unassigned")

    # For each CC200 ROI label 1..200, compute centroid and read Schaefer.
    networks: List[str] = []
    n_unassigned = 0
    for cc_label in range(1, N_ROIS + 1):
        mask = cc_data == cc_label
        if not mask.any():
            networks.append("Unassigned")
            n_unassigned += 1
            continue

        # Centroid voxel of this CC200 ROI.
        com = ndi.center_of_mass(mask)
        ci, cj, ck = (int(round(c)) for c in com)
        ci = np.clip(ci, 0, sch_data.shape[0] - 1)
        cj = np.clip(cj, 0, sch_data.shape[1] - 1)
        ck = np.clip(ck, 0, sch_data.shape[2] - 1)

        sch_at_centroid = int(sch_data[ci, cj, ck])
        if sch_at_centroid == 0:
            # Centroid landed in unlabelled tissue (white matter / outside
            # cortex). Fall back to majority-vote of Schaefer labels found
            # anywhere inside the CC200 mask.
            vals = sch_data[mask]
            vals = vals[vals > 0]
            if vals.size:
                sch_at_centroid = int(np.bincount(vals).argmax())

        net = label_to_net.get(sch_at_centroid, "Unassigned")
        networks.append(net)
        if net == "Unassigned":
            n_unassigned += 1

    counts = pd.Series(networks).value_counts().reindex(NETWORK_ORDER, fill_value=0)
    print("      network counts (in CC200):")
    for k, v in counts.items():
        print(f"        {k:<15s} {v:3d}")
    return networks


# --------------------------------------------------------------------------- #
# Connectivity                                                                #
# --------------------------------------------------------------------------- #


def load_group_means() -> Dict[str, np.ndarray]:
    """Re-fetch the 50-subject sample and return mean ASD / Control matrices."""
    print("[2/4] Loading 50-subject sample + computing group-mean matrices ...")
    bunch = datasets.fetch_abide_pcp(
        data_dir=str(DATA_DIR), pipeline="cpac", derivatives=["rois_cc200"],
        band_pass_filtering=True, global_signal_regression=False,
        quality_checked=True, n_subjects=50, verbose=0,
    )
    pheno = pd.DataFrame(bunch.phenotypic)
    pheno.columns = [str(c) for c in pheno.columns]
    pheno["DX_GROUP"] = pheno["DX_GROUP"].astype(int)
    groups = pheno["DX_GROUP"].map(DX_LABELS).values

    by_group: Dict[str, List[np.ndarray]] = {"ASD": [], "Control": []}
    for ts, grp in zip(bunch.rois_cc200, groups):
        arr = ts if isinstance(ts, np.ndarray) else np.loadtxt(ts)
        if arr.ndim != 2 or arr.shape[1] != N_ROIS:
            continue
        c = np.nan_to_num(np.corrcoef(arr.T))
        if grp in by_group:
            by_group[grp].append(c)
    means = {g: np.mean(np.stack(m), axis=0) for g, m in by_group.items()}
    print(f"      ASD n={len(by_group['ASD'])}, Control n={len(by_group['Control'])}")
    return means


# --------------------------------------------------------------------------- #
# Sort + plot                                                                 #
# --------------------------------------------------------------------------- #


def network_permutation(networks: List[str]) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """Return (perm, blocks) — perm reorders ROIs by network; blocks is a
    list of ``(name, start_idx, end_idx_exclusive)`` for drawing dividers."""
    order_rank = {n: i for i, n in enumerate(NETWORK_ORDER)}
    keys = [(order_rank.get(n, len(NETWORK_ORDER)), i) for i, n in enumerate(networks)]
    perm = np.array([i for _, i in sorted(keys)])
    sorted_nets = [networks[i] for i in perm]

    blocks: List[Tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(sorted_nets) + 1):
        if i == len(sorted_nets) or sorted_nets[i] != sorted_nets[start]:
            blocks.append((sorted_nets[start], start, i))
            start = i
    return perm, blocks


def _draw_network_grid(ax, blocks):
    """Draw white separators + center labels on both axes of a heatmap."""
    n = blocks[-1][2]
    for _, _, end in blocks[:-1]:
        ax.axhline(end - 0.5, color="white", lw=1.4)
        ax.axvline(end - 0.5, color="white", lw=1.4)

    # Network names along x (rotated) and y (horizontal).
    for name, s, e in blocks:
        mid = (s + e) / 2.0
        color = NETWORK_COLORS.get(name, "#444")
        ax.text(mid, n + 4, name, rotation=45, ha="right", va="top",
                fontsize=8, color=color, weight="bold")
        ax.text(-4, mid, name, rotation=0, ha="right", va="center",
                fontsize=8, color=color, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def figure_sorted(means: Dict[str, np.ndarray], networks: List[str]) -> None:
    print("[3/4] Plotting network-sorted comparison ...")
    perm, blocks = network_permutation(networks)
    asd_s = means["ASD"][perm][:, perm]
    ctl_s = means["Control"][perm][:, perm]
    diff_s = asd_s - ctl_s
    vmax_diff = float(np.max(np.abs(diff_s))) or 1e-3

    # ---------- matplotlib ---------- #
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    panels = [
        ("Mean ASD", asd_s, -1, 1, "RdBu_r", "mean r"),
        ("Mean Control", ctl_s, -1, 1, "RdBu_r", "mean r"),
        ("Difference (ASD − Control)", diff_s, -vmax_diff, vmax_diff,
         "RdBu_r", "Δ r"),
    ]
    for ax, (title, mat, vmin, vmax, cmap, cbar_label) in zip(axes, panels):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title, pad=10)
        _draw_network_grid(ax, blocks)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    # Add a one-line legend reminding the reader of the 7-network color code.
    handles = [mpatches.Patch(color=NETWORK_COLORS[n], label=n)
               for n in NETWORK_ORDER if any(b[0] == n for b in blocks)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(
        "ABIDE I — group-mean functional connectivity, sorted by Yeo-7 network",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = VIS_DIR / "connectivity_comparison_networks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      [png ] {out}")

    # ---------- plotly (interactive) ---------- #
    pfig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Mean ASD", "Mean Control",
                        "Difference (ASD − Control)"),
        horizontal_spacing=0.07,
    )
    n = asd_s.shape[0]
    pfig.add_trace(go.Heatmap(z=asd_s, zmin=-1, zmax=1, colorscale="RdBu",
                              reversescale=True, showscale=False),
                   row=1, col=1)
    pfig.add_trace(go.Heatmap(z=ctl_s, zmin=-1, zmax=1, colorscale="RdBu",
                              reversescale=True, showscale=False),
                   row=1, col=2)
    pfig.add_trace(go.Heatmap(z=diff_s, zmin=-vmax_diff, zmax=vmax_diff,
                              colorscale="RdBu", reversescale=True,
                              colorbar=dict(title="Δ r")),
                   row=1, col=3)

    # Network grid lines as shapes; labels as annotations.
    for col in (1, 2, 3):
        xref = f"x{col}" if col > 1 else "x"
        yref = f"y{col}" if col > 1 else "y"
        for _, _, end in blocks[:-1]:
            pfig.add_shape(type="line",
                           x0=end - 0.5, x1=end - 0.5, y0=-0.5, y1=n - 0.5,
                           line=dict(color="white", width=1.4),
                           xref=xref, yref=yref)
            pfig.add_shape(type="line",
                           y0=end - 0.5, y1=end - 0.5, x0=-0.5, x1=n - 0.5,
                           line=dict(color="white", width=1.4),
                           xref=xref, yref=yref)
        for name, s, e in blocks:
            mid = (s + e) / 2.0
            color = NETWORK_COLORS.get(name, "#444")
            pfig.add_annotation(
                x=mid, y=-3, text=name, textangle=-45,
                showarrow=False, font=dict(size=9, color=color),
                xref=xref, yref=yref, yanchor="bottom",
            )

    for col in (1, 2, 3):
        pfig.update_yaxes(autorange="reversed", showticklabels=False,
                          row=1, col=col)
        pfig.update_xaxes(showticklabels=False, row=1, col=col)
    pfig.update_layout(
        title="ABIDE I — connectivity sorted by Yeo-7 network",
        width=1500, height=600, margin=dict(t=80, b=80),
    )
    out = VIS_DIR / "connectivity_comparison_networks.html"
    pfig.write_html(out, include_plotlyjs="cdn")
    print(f"      [html] {out}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    networks = build_cc200_to_network()
    means = load_group_means()
    figure_sorted(means, networks)
    print("[4/4] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
