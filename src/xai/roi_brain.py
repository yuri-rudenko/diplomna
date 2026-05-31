"""
Project ROI importance onto the brain via nilearn and the CC200 atlas.

Usage:
    python -m src.xai.roi_brain --model attention_vae --fold best
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
DATA_DIR = PROJECT_DIR / "data"


def _get_cc200_atlas():
    """Fetch CC200 atlas NIfTI via nilearn (cached after first download).

    Uses the SSL-robust fetch wrapper: nitrc.org (the Craddock host) ships an
    invalid certificate, so a plain fetch fails. robust_fetch clears any
    half-written cache and retries with TLS verification disabled.
    """
    from nilearn import datasets

    from src.utils.atlas_fetch import robust_fetch

    cdir = DATA_DIR / "craddock_2012"
    atlas = robust_fetch(
        datasets.fetch_atlas_craddock_2012,
        data_dir=str(cdir),
        partial_dirs=[cdir],
    )
    # The scorr_mean parcellation is the 200-ROI version used in ABIDE
    return atlas.scorr_mean


def roi_importance_to_brain_img(roi_importance: np.ndarray, atlas_img_path: str):
    """
    Map (200,) ROI importance values onto CC200 atlas voxels.

    Returns a NIfTI stat map with voxel values = importance of the ROI it belongs to.
    """
    import nibabel as nib
    import numpy as np

    atlas_img = nib.load(atlas_img_path)
    atlas_data = atlas_img.get_fdata()
    stat_data = np.zeros_like(atlas_data, dtype=np.float32)

    for roi_idx in range(len(roi_importance)):
        label = roi_idx + 1  # CC200 labels are 1-indexed
        stat_data[atlas_data == label] = roi_importance[roi_idx]

    return nib.Nifti1Image(stat_data, atlas_img.affine, atlas_img.header)


def plot_brain_importance(
    roi_importance: np.ndarray,
    model_type: str = "attention_vae",
    fold_idx: int = 0,
    top_n: int = 20,
) -> None:
    """
    Generate glass-brain and stat-map plots of ROI importance.
    Saves PNG figures to results/figures/.
    """
    from nilearn import plotting

    atlas_path = _get_cc200_atlas()
    stat_img = roi_importance_to_brain_img(roi_importance, atlas_path)

    tag = f"{model_type}_fold{fold_idx}"
    out_glass = str(FIGURES_DIR / f"brain_glassbrain_{tag}.png")
    out_stat = str(FIGURES_DIR / f"brain_statmap_{tag}.png")

    # Glass brain: shows top activated ROIs
    display = plotting.plot_glass_brain(
        stat_img,
        colorbar=True,
        title=f"ROI importance (SHAP) — {model_type}",
        display_mode="lzr",
        plot_abs=False,
        output_file=out_glass,
    )
    print(f"  Saved glass brain → {out_glass}")

    # Stat map on MNI template
    display2 = plotting.plot_stat_map(
        stat_img,
        title=f"SHAP ROI importance — {model_type}",
        colorbar=True,
        output_file=out_stat,
    )
    print(f"  Saved stat map    → {out_stat}")


def plot_attention_heatmap(
    model_type: str = "attention_vae",
    fold_idx: int = 0,
    exp_name: str | None = None,
) -> None:
    """
    Plot mean attention weights from the PatchAttention block.
    Loads a checkpoint and runs a few test samples through it.
    """
    import matplotlib.pyplot as plt
    import torch

    from src.data.download_abide import load_processed
    from src.data.preprocessing import fit_scaler, apply_scaler, vectorize
    from src.data.splits import load_splits
    from src.models.attention_vae import AttentionVAE
    from src.utils.checkpoints import resolve_checkpoint

    if model_type != "attention_vae":
        print("Attention heatmap only available for attention_vae")
        return

    device = torch.device("cpu")
    X_raw, pheno = load_processed()
    folds = load_splits()
    _, test_idx = folds[fold_idx]

    X_test_vec = vectorize(X_raw[test_idx])
    scaler = fit_scaler(vectorize(X_raw[folds[fold_idx][0]]))
    X_test_vec, = apply_scaler(scaler, X_test_vec, X_test_vec)[1:2]  # only test

    model = AttentionVAE().to(device)
    ckpt = resolve_checkpoint("attention_vae", fold_idx, exp_name)
    if ckpt is None:
        print(f"Checkpoint not found for attention_vae fold {fold_idx} (exp={exp_name})")
        return

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(X_test_vec[:32])
        model(x)
        weights = model.attention_weights  # (B, n_patches, n_patches)

    if weights is None:
        print("No attention weights captured")
        return

    mean_attn = weights.mean(dim=0).numpy()  # (n_patches, n_patches)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_attn, cmap="viridis", vmin=0)
    ax.set_title(f"Mean patch attention weights\n{model_type} — fold {fold_idx}")
    ax.set_xlabel("Key patch")
    ax.set_ylabel("Query patch")
    plt.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    out = FIGURES_DIR / f"attention_heatmap_{model_type}_fold{fold_idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention heatmap → {out}")


if __name__ == "__main__":
    from src.utils.checkpoints import resolve_fold

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="attention_vae")
    parser.add_argument("--fold", default="0",
                        help="Fold index or 'best' (default: 0)")
    parser.add_argument("--exp", default=None,
                        help="Experiment name from run_experiments.py (e.g. aug_full). "
                             "Resolves checkpoints from results/experiments/{exp}/checkpoints/")
    args = parser.parse_args()

    fold_idx = resolve_fold(args.fold, args.model, args.exp)

    imp_path = FIGURES_DIR / f"roi_importance_{args.model}_fold{fold_idx}.npy"
    if not imp_path.exists():
        print(f"ROI importance file not found: {imp_path}")
        print("Run src.xai.shap_explain (or src.xai.run_xai) first.")
    else:
        roi_importance = np.load(imp_path)
        plot_brain_importance(roi_importance, args.model, fold_idx)

    if args.model == "attention_vae":
        plot_attention_heatmap(args.model, fold_idx, exp_name=args.exp)
