"""
Sample-data visualization for the ABIDE I diploma project.

Downloads a small (n=50) sample of preprocessed ABIDE I resting-state fMRI
ROI time-series via nilearn (CPAC pipeline, Craddock-200 atlas) and
produces five exploratory figures used to sanity-check the data and
illustrate the dataset in the diploma write-up:

    a) subject overview      (group counts, age, sex)
    b) connectivity_single   (200x200 correlation matrix for one subject)
    c) connectivity_comparison (mean ASD vs. mean Control + difference)
    d) timeseries            (BOLD of 5 ROIs, ASD vs. Control example)
    e) embedding_2d          (PCA + UMAP scatter colored by diagnosis)

Each figure is saved as both a static PNG and an interactive Plotly HTML
into the project's ``visualizations/`` directory.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

RANDOM_STATE = 42
N_SUBJECTS = 50
N_ROIS = 200

# Directory layout (resolved relative to this file so the script works
# regardless of the cwd it is launched from).
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
VIS_DIR = PROJECT_DIR / "visualizations"

# Diagnosis encoding used by ABIDE: DX_GROUP == 1 means ASD, 2 means TC.
DX_LABELS = {1: "ASD", 2: "Control"}
SEX_LABELS = {1: "Male", 2: "Female"}

np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #


def download_abide():
    """Fetch a 50-subject CC200 sample of ABIDE I (CPAC pipeline)."""
    from nilearn import datasets

    print(f"[1/6] Downloading ABIDE I sample (n_subjects={N_SUBJECTS}) ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bunch = datasets.fetch_abide_pcp(
        data_dir=str(DATA_DIR),
        pipeline="cpac",
        derivatives=["rois_cc200"],
        band_pass_filtering=True,
        global_signal_regression=False,
        quality_checked=True,
        n_subjects=N_SUBJECTS,
        verbose=1,
    )
    print(f"      done — {len(bunch.rois_cc200)} ROI files available")
    return bunch


def load_phenotypic(bunch) -> pd.DataFrame:
    """Return a tidy phenotypic table with readable group / sex labels."""
    pheno = pd.DataFrame(bunch.phenotypic)
    # nilearn returns a structured array; ensure column names are strings.
    pheno.columns = [str(c) for c in pheno.columns]

    keep = [c for c in ("SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX") if c in pheno.columns]
    pheno = pheno[keep].copy()

    pheno["DX_GROUP"] = pheno["DX_GROUP"].astype(int)
    pheno["group"] = pheno["DX_GROUP"].map(DX_LABELS)
    if "SEX" in pheno.columns:
        pheno["sex"] = pheno["SEX"].astype(int).map(SEX_LABELS)
    if "AGE_AT_SCAN" in pheno.columns:
        pheno["age"] = pd.to_numeric(pheno["AGE_AT_SCAN"], errors="coerce")
    return pheno.reset_index(drop=True)


def load_timeseries(bunch) -> Tuple[List[np.ndarray], List[int]]:
    """Collect each subject's CC200 ROI time-series.

    In recent nilearn (>= 0.10) ``fetch_abide_pcp`` already returns the
    derivative as in-memory ``(T, 200)`` numpy arrays. For robustness we
    also accept legacy file-path entries (older nilearn versions).
    """
    print(f"[2/6] Loading ROI time-series ...")
    ts_list: List[np.ndarray] = []
    ok_indices: List[int] = []
    for i, item in enumerate(bunch.rois_cc200):
        try:
            if isinstance(item, np.ndarray):
                arr = item
            else:
                arr = np.loadtxt(item)
            if arr.ndim != 2 or arr.shape[1] != N_ROIS:
                raise ValueError(f"unexpected shape {arr.shape}")
            ts_list.append(np.asarray(arr, dtype=np.float64))
            ok_indices.append(i)
        except Exception as exc:  # noqa: BLE001 — per-subject fault tolerance
            print(f"      [skip] subject idx {i}: {exc}")
    print(f"      loaded {len(ts_list)} / {len(bunch.rois_cc200)} subjects")
    return ts_list, ok_indices


def compute_connectivity(ts: np.ndarray) -> np.ndarray:
    """Pearson-correlation functional-connectivity matrix (200x200)."""
    mat = np.corrcoef(ts.T)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Saving helpers                                                              #
# --------------------------------------------------------------------------- #


def _save_matplotlib(fig, name: str) -> None:
    out = VIS_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      [png ] {out}")


def _save_plotly(fig, name: str) -> None:
    out = VIS_DIR / f"{name}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"      [html] {out}")


# --------------------------------------------------------------------------- #
# Figure (a) — subject overview                                               #
# --------------------------------------------------------------------------- #


def figure_subject_overview(pheno: pd.DataFrame) -> None:
    """Group count + age histogram + sex distribution."""
    print("[3/6] Figure (a): subject overview ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Group counts
    counts = pheno["group"].value_counts().reindex(["ASD", "Control"]).fillna(0)
    axes[0].bar(counts.index, counts.values, color=["#d62728", "#1f77b4"])
    axes[0].set_title("Subject count")
    axes[0].set_ylabel("subjects")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.3, str(int(v)), ha="center")

    # Age histogram
    if "age" in pheno.columns:
        for grp, color in zip(["ASD", "Control"], ["#d62728", "#1f77b4"]):
            ages = pheno.loc[pheno["group"] == grp, "age"].dropna().values
            if len(ages):
                axes[1].hist(ages, bins=15, alpha=0.6, label=grp, color=color)
        axes[1].set_title("Age distribution")
        axes[1].set_xlabel("age (years)")
        axes[1].legend()

    # Sex distribution (grouped bar)
    if "sex" in pheno.columns:
        cross = pd.crosstab(pheno["sex"], pheno["group"]).reindex(
            index=["Male", "Female"], columns=["ASD", "Control"], fill_value=0
        )
        x = np.arange(len(cross.index))
        w = 0.35
        axes[2].bar(x - w / 2, cross["ASD"], w, label="ASD", color="#d62728")
        axes[2].bar(x + w / 2, cross["Control"], w, label="Control", color="#1f77b4")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(cross.index)
        axes[2].set_title("Sex distribution")
        axes[2].legend()

    fig.tight_layout()
    _save_matplotlib(fig, "subject_overview")

    # Plotly mirror
    pfig = make_subplots(rows=1, cols=3, subplot_titles=("Subject count", "Age", "Sex"))
    pfig.add_trace(
        go.Bar(x=counts.index.tolist(), y=counts.values.tolist(),
               marker_color=["#d62728", "#1f77b4"], showlegend=False),
        row=1, col=1,
    )
    if "age" in pheno.columns:
        for grp, color in zip(["ASD", "Control"], ["#d62728", "#1f77b4"]):
            ages = pheno.loc[pheno["group"] == grp, "age"].dropna().values
            pfig.add_trace(
                go.Histogram(x=ages, name=grp, marker_color=color, opacity=0.6),
                row=1, col=2,
            )
        pfig.update_layout(barmode="overlay")
    if "sex" in pheno.columns:
        cross = pd.crosstab(pheno["sex"], pheno["group"]).reindex(
            index=["Male", "Female"], columns=["ASD", "Control"], fill_value=0
        )
        pfig.add_trace(
            go.Bar(x=cross.index.tolist(), y=cross["ASD"].tolist(),
                   name="ASD", marker_color="#d62728"),
            row=1, col=3,
        )
        pfig.add_trace(
            go.Bar(x=cross.index.tolist(), y=cross["Control"].tolist(),
                   name="Control", marker_color="#1f77b4"),
            row=1, col=3,
        )
    pfig.update_layout(title="ABIDE I sample — subject overview", height=420)
    _save_plotly(pfig, "subject_overview")


# --------------------------------------------------------------------------- #
# Figure (b) — single-subject connectivity                                    #
# --------------------------------------------------------------------------- #


def figure_connectivity_single(
    pheno: pd.DataFrame, ts_list: List[np.ndarray], ok_indices: List[int]
) -> np.ndarray:
    """Plot one subject's 200x200 functional-connectivity matrix."""
    print("[4/6] Figure (b): single-subject connectivity ...")
    # Pick the first ASD subject we have time-series for.
    pheno_ok = pheno.iloc[ok_indices].reset_index(drop=True)
    asd_pos = pheno_ok.index[pheno_ok["group"] == "ASD"]
    pick_local = int(asd_pos[0]) if len(asd_pos) else 0
    label = pheno_ok.loc[pick_local, "group"]
    sub_id = int(pheno_ok.loc[pick_local, "SUB_ID"])
    conn = compute_connectivity(ts_list[pick_local])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(conn, vmin=-1, vmax=1, cmap="RdBu_r", square=True,
                cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title(f"Connectivity — subject {sub_id} ({label})")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    _save_matplotlib(fig, "connectivity_single")

    pfig = go.Figure(
        data=go.Heatmap(z=conn, zmin=-1, zmax=1, colorscale="RdBu",
                        reversescale=True, colorbar=dict(title="Pearson r"))
    )
    pfig.update_layout(
        title=f"Connectivity — subject {sub_id} ({label})",
        xaxis_title="ROI", yaxis_title="ROI",
        width=720, height=680, yaxis_autorange="reversed",
    )
    _save_plotly(pfig, "connectivity_single")
    return conn


# --------------------------------------------------------------------------- #
# Figure (c) — group comparison                                               #
# --------------------------------------------------------------------------- #


def figure_connectivity_comparison(
    pheno: pd.DataFrame, ts_list: List[np.ndarray], ok_indices: List[int]
) -> Dict[str, np.ndarray]:
    """Mean connectivity for ASD, mean for Control, and their difference."""
    print("[5/6] Figure (c): group connectivity comparison ...")
    pheno_ok = pheno.iloc[ok_indices].reset_index(drop=True)
    by_group: Dict[str, List[np.ndarray]] = {"ASD": [], "Control": []}
    for i, row in pheno_ok.iterrows():
        grp = row["group"]
        if grp in by_group:
            by_group[grp].append(compute_connectivity(ts_list[i]))

    means = {g: np.mean(np.stack(m), axis=0) if m else np.zeros((N_ROIS, N_ROIS))
             for g, m in by_group.items()}
    diff = means["ASD"] - means["Control"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (title, mat) in zip(
        axes[:2], [("Mean ASD", means["ASD"]), ("Mean Control", means["Control"])]
    ):
        sns.heatmap(mat, vmin=-1, vmax=1, cmap="RdBu_r", square=True,
                    cbar_kws={"label": "mean r"}, ax=ax)
        ax.set_title(f"{title}  (n={len(by_group[title.split()[1]])})")
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")

    vmax = float(np.max(np.abs(diff))) or 1e-3
    sns.heatmap(diff, vmin=-vmax, vmax=vmax, cmap="RdBu_r", square=True,
                cbar_kws={"label": "Δ r (ASD − Control)"}, ax=axes[2])
    axes[2].set_title("Difference (ASD − Control)")
    axes[2].set_xlabel("ROI")
    axes[2].set_ylabel("ROI")
    fig.tight_layout()
    _save_matplotlib(fig, "connectivity_comparison")

    pfig = make_subplots(rows=1, cols=3,
                         subplot_titles=("Mean ASD", "Mean Control",
                                         "Difference (ASD − Control)"))
    pfig.add_trace(go.Heatmap(z=means["ASD"], zmin=-1, zmax=1,
                              colorscale="RdBu", reversescale=True,
                              showscale=False), row=1, col=1)
    pfig.add_trace(go.Heatmap(z=means["Control"], zmin=-1, zmax=1,
                              colorscale="RdBu", reversescale=True,
                              showscale=False), row=1, col=2)
    pfig.add_trace(go.Heatmap(z=diff, zmin=-vmax, zmax=vmax,
                              colorscale="RdBu", reversescale=True,
                              colorbar=dict(title="Δ r")), row=1, col=3)
    for r in range(1, 2):
        for c in range(1, 4):
            pfig.update_yaxes(autorange="reversed", row=r, col=c)
    pfig.update_layout(title="ABIDE I — average connectivity comparison",
                       width=1400, height=520)
    _save_plotly(pfig, "connectivity_comparison")
    return {"asd_mean": means["ASD"], "control_mean": means["Control"], "diff": diff}


# --------------------------------------------------------------------------- #
# Figure (d) — time-series                                                    #
# --------------------------------------------------------------------------- #


def figure_timeseries(
    pheno: pd.DataFrame, ts_list: List[np.ndarray], ok_indices: List[int]
) -> None:
    """BOLD signal over time for 5 ROIs in one ASD and one Control subject."""
    print("[6/6] Figure (d): example time-series ...")
    pheno_ok = pheno.iloc[ok_indices].reset_index(drop=True)
    rois = [0, 50, 100, 150, 199]
    # Note: CC200 ROIs are functionally derived parcels with no canonical
    # anatomical labels; we use index-based names for clarity.
    roi_names = [f"ROI {r}" for r in rois]

    asd_pos = pheno_ok.index[pheno_ok["group"] == "ASD"]
    tc_pos = pheno_ok.index[pheno_ok["group"] == "Control"]
    if len(asd_pos) == 0 or len(tc_pos) == 0:
        print("      [skip] need at least one subject from each group")
        return
    asd_idx = int(asd_pos[0])
    tc_idx = int(tc_pos[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    palette = sns.color_palette("tab10", n_colors=len(rois))
    for ax, idx, label, color in zip(
        axes, [asd_idx, tc_idx], ["ASD", "Control"], ["#d62728", "#1f77b4"]
    ):
        ts = ts_list[idx]
        sub_id = int(pheno_ok.loc[idx, "SUB_ID"])
        for j, r in enumerate(rois):
            ax.plot(ts[:, r], color=palette[j], lw=1, label=roi_names[j])
        ax.set_title(f"{label} subject {sub_id} — BOLD time-series")
        ax.set_xlabel("time (TR)")
        ax.set_ylabel("BOLD (a.u.)")
        ax.legend(ncol=5, fontsize=8, loc="upper right")
        ax.set_facecolor((1, 1, 1))
        for spine in ax.spines.values():
            spine.set_color(color)
    fig.tight_layout()
    _save_matplotlib(fig, "timeseries")

    pfig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"ASD subject {int(pheno_ok.loc[asd_idx, 'SUB_ID'])} — BOLD",
            f"Control subject {int(pheno_ok.loc[tc_idx, 'SUB_ID'])} — BOLD",
        ),
    )
    for row, idx in enumerate([asd_idx, tc_idx], start=1):
        ts = ts_list[idx]
        for j, r in enumerate(rois):
            pfig.add_trace(
                go.Scatter(y=ts[:, r], mode="lines", name=roi_names[j],
                           legendgroup=roi_names[j],
                           showlegend=(row == 1)),
                row=row, col=1,
            )
    pfig.update_xaxes(title_text="time (TR)", row=2, col=1)
    pfig.update_yaxes(title_text="BOLD", row=1, col=1)
    pfig.update_yaxes(title_text="BOLD", row=2, col=1)
    pfig.update_layout(title="Example BOLD time-series — ASD vs. Control",
                       height=700, width=1100)
    _save_plotly(pfig, "timeseries")


# --------------------------------------------------------------------------- #
# Figure (e) — 2D embedding                                                   #
# --------------------------------------------------------------------------- #


def figure_embedding_2d(
    pheno: pd.DataFrame, ts_list: List[np.ndarray], ok_indices: List[int]
) -> None:
    """PCA + UMAP scatter of upper-triangular connectivity vectors."""
    print("[+/+] Figure (e): 2D embedding ...")
    pheno_ok = pheno.iloc[ok_indices].reset_index(drop=True)
    iu = np.triu_indices(N_ROIS, k=1)
    X = np.stack([compute_connectivity(ts)[iu] for ts in ts_list])
    y = pheno_ok["group"].values

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    umap_pts = None
    try:
        import umap  # type: ignore

        # Cap n_neighbors so UMAP works on tiny samples.
        n_neighbors = max(2, min(15, X.shape[0] - 1))
        umap_pts = umap.UMAP(
            n_components=2, random_state=RANDOM_STATE,
            n_neighbors=n_neighbors,
        ).fit_transform(X)
    except Exception as exc:  # noqa: BLE001
        print(f"      [warn] UMAP failed: {exc} — only PCA panel will be drawn")

    cols = 2 if umap_pts is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), squeeze=False)
    color_map = {"ASD": "#d62728", "Control": "#1f77b4"}

    def _scatter(ax, pts, title):
        for grp, color in color_map.items():
            mask = y == grp
            ax.scatter(pts[mask, 0], pts[mask, 1], c=color, label=grp,
                       alpha=0.8, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.legend()

    _scatter(axes[0, 0], pca, "PCA (2 components)")
    if umap_pts is not None:
        _scatter(axes[0, 1], umap_pts, "UMAP (2 components)")
    fig.tight_layout()
    _save_matplotlib(fig, "embedding_2d")

    pfig = make_subplots(
        rows=1, cols=cols,
        subplot_titles=("PCA", "UMAP")[:cols],
    )
    for col_idx, (pts, name) in enumerate(
        [(pca, "PCA")] + ([(umap_pts, "UMAP")] if umap_pts is not None else []),
        start=1,
    ):
        for grp, color in color_map.items():
            mask = y == grp
            pfig.add_trace(
                go.Scatter(
                    x=pts[mask, 0], y=pts[mask, 1], mode="markers",
                    marker=dict(color=color, size=10,
                                line=dict(color="white", width=1)),
                    name=f"{name} — {grp}",
                    legendgroup=grp, showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )
    pfig.update_layout(
        title="2D embedding of subject connectivity vectors",
        width=600 * cols, height=520,
    )
    _save_plotly(pfig, "embedding_2d")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    bunch = download_abide()
    pheno = load_phenotypic(bunch)
    ts_list, ok_indices = load_timeseries(bunch)
    if not ts_list:
        print("ERROR: no time-series could be loaded — aborting", file=sys.stderr)
        return 1

    pheno_ok = pheno.iloc[ok_indices].reset_index(drop=True)

    # Run each figure independently so one failure doesn't abort the rest.
    for fn, args in [
        (figure_subject_overview, (pheno_ok,)),
        (figure_connectivity_single, (pheno, ts_list, ok_indices)),
        (figure_connectivity_comparison, (pheno, ts_list, ok_indices)),
        (figure_timeseries, (pheno, ts_list, ok_indices)),
        (figure_embedding_2d, (pheno, ts_list, ok_indices)),
    ]:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"      [fail] {fn.__name__}: {exc}")
            traceback.print_exc()

    # Summary
    counts = pheno_ok["group"].value_counts().to_dict()
    sample_shape = ts_list[0].shape
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subjects downloaded     : {len(bunch.rois_cc200)}")
    print(f"Subjects usable (loaded): {len(ts_list)}")
    print(f"  ASD                   : {counts.get('ASD', 0)}")
    print(f"  Control               : {counts.get('Control', 0)}")
    print(f"Per-subject TS shape    : {sample_shape}  (timepoints x ROIs)")
    print(f"Data cache              : {DATA_DIR}")
    print(f"Visualizations          : {VIS_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
