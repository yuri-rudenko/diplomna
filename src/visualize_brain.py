"""
Brain-scan / atlas visualization for the ABIDE I diploma project.

Complements ``visualize_sample.py`` (which works on ROI time-series) by
showing the *spatial* structure of the data:

    a) atlas_cc200       — the 200 functional ROIs overlaid on the
                           standard MNI152 template (this is the
                           parcellation our CC200 time-series come from)
    b) anat_subject      — anatomical (T1) view of one ABIDE subject
                           (if available) or the MNI template fallback
    c) func_mean         — mean BOLD image (across time) of one ABIDE
                           subject, multi-slice axial montage
    d) func_glass        — glass-brain projection of the same mean BOLD
    e) func_view         — interactive 3D HTML viewer of mean BOLD
    f) atlas_view        — interactive 3D HTML viewer of the CC200 atlas

Static figures are saved as PNG, interactive views as HTML, into
``visualizations/``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting

RANDOM_STATE = 42

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
VIS_DIR = PROJECT_DIR / "visualizations"

np.random.seed(RANDOM_STATE)


def _save_fig(fig, name: str) -> None:
    out = VIS_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      [png ] {out}")


def _save_display(display, name: str) -> None:
    """Save a nilearn ``OrthoSlicer`` / projector and close it."""
    out = VIS_DIR / f"{name}.png"
    display.savefig(out, dpi=150)
    display.close()
    print(f"      [png ] {out}")


def _save_view(view, name: str) -> None:
    """Save a nilearn interactive HTML view."""
    out = VIS_DIR / f"{name}.html"
    view.save_as_html(out)
    print(f"      [html] {out}")


# --------------------------------------------------------------------------- #
# Atlas figures                                                               #
# --------------------------------------------------------------------------- #


def fetch_cc200_atlas():
    """Return the Craddock-200 functional atlas as a 3D NIfTI image.

    ``fetch_atlas_craddock_2012`` ships several parcellations bundled into a
    single 4D NIfTI. We extract volume index 19, which corresponds to the
    *scorr_2level / 200-cluster* solution used by ABIDE PCP's
    ``rois_cc200`` derivative.
    """
    print("[1/6] Fetching Craddock-200 atlas ...")
    craddock = datasets.fetch_atlas_craddock_2012(data_dir=str(DATA_DIR))
    # The bundled 4D file (`maps`) contains parcellations at multiple K
    # values; volume 19 is the ~200-cluster solution that matches ABIDE
    # PCP's ``rois_cc200`` derivative.
    atlas_img = image.index_img(craddock["maps"], 19)
    print(f"      atlas shape: {atlas_img.shape}")
    return atlas_img


def figure_atlas_cc200(atlas_img) -> None:
    print("[2/6] Figure: CC200 atlas on MNI152 ...")
    disp = plotting.plot_roi(
        atlas_img,
        bg_img=datasets.load_mni152_template(),
        title="Craddock-200 atlas (200 functional ROIs) on MNI152",
        display_mode="ortho",
        draw_cross=False,
        cmap="tab20",
        alpha=0.7,
    )
    _save_display(disp, "atlas_cc200")

    # Multi-slice axial montage for a denser overview.
    fig = plt.figure(figsize=(16, 4))
    disp = plotting.plot_roi(
        atlas_img,
        bg_img=datasets.load_mni152_template(),
        display_mode="z",
        cut_coords=8,
        cmap="tab20",
        alpha=0.7,
        title="CC200 — axial slices",
        figure=fig,
    )
    out = VIS_DIR / "atlas_cc200_axial.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    disp.close()
    print(f"      [png ] {out}")


def figure_atlas_view(atlas_img) -> None:
    print("[3/6] Interactive: CC200 atlas viewer ...")
    view = plotting.view_img(
        atlas_img,
        bg_img=datasets.load_mni152_template(),
        title="Craddock-200 atlas — interactive viewer",
        cmap="tab20",
        symmetric_cmap=False,
        opacity=0.7,
    )
    _save_view(view, "atlas_cc200_view")


# --------------------------------------------------------------------------- #
# Sample subject — actual fMRI scan                                           #
# --------------------------------------------------------------------------- #


def fetch_one_func():
    """Download the preprocessed 4D BOLD of a single ABIDE subject."""
    print("[4/6] Fetching one subject's func_preproc 4D BOLD ...")
    bunch = datasets.fetch_abide_pcp(
        data_dir=str(DATA_DIR),
        pipeline="cpac",
        derivatives=["func_preproc"],
        band_pass_filtering=True,
        global_signal_regression=False,
        quality_checked=True,
        n_subjects=1,
        verbose=1,
    )
    func = bunch.func_preproc[0]
    if not isinstance(func, (str, Path)):
        # Newer nilearn returns numpy arrays — wrap as Nifti1Image.
        # But func_preproc is usually still returned as a file path; defend.
        func_img = nib.Nifti1Image(np.asarray(func), affine=np.eye(4))
    else:
        func_img = nib.load(func)
    pheno = bunch.phenotypic
    sub_id = int(pheno["SUB_ID"].iloc[0] if hasattr(pheno["SUB_ID"], "iloc")
                 else pheno["SUB_ID"][0])
    dx = int(pheno["DX_GROUP"].iloc[0] if hasattr(pheno["DX_GROUP"], "iloc")
             else pheno["DX_GROUP"][0])
    label = "ASD" if dx == 1 else "Control"
    print(f"      subject {sub_id} ({label}) — 4D shape: {func_img.shape}")
    return func_img, sub_id, label


def figure_func_mean(func_img, sub_id: int, label: str):
    print("[5/6] Figures: mean BOLD slices + glass brain ...")
    mean_img = image.mean_img(func_img, copy_header=True)

    # Multi-slice axial montage of the mean BOLD.
    fig = plt.figure(figsize=(16, 4))
    disp = plotting.plot_epi(
        mean_img,
        display_mode="z",
        cut_coords=8,
        title=f"Mean BOLD (axial) — subject {sub_id} ({label})",
        figure=fig,
    )
    out = VIS_DIR / "func_mean_axial.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    disp.close()
    print(f"      [png ] {out}")

    # Ortho view (sagittal + coronal + axial through one cut).
    disp = plotting.plot_epi(
        mean_img,
        display_mode="ortho",
        title=f"Mean BOLD (ortho) — subject {sub_id} ({label})",
    )
    _save_display(disp, "func_mean_ortho")

    # Glass-brain projection.
    disp = plotting.plot_glass_brain(
        mean_img,
        title=f"Mean BOLD glass brain — subject {sub_id} ({label})",
        display_mode="ortho",
        colorbar=True,
    )
    _save_display(disp, "func_glass")

    return mean_img


def figure_func_view(mean_img, sub_id: int, label: str) -> None:
    print("[6/6] Interactive: mean BOLD 3D viewer ...")
    view = plotting.view_img(
        mean_img,
        title=f"Mean BOLD — subject {sub_id} ({label})",
        symmetric_cmap=False,
        cmap="magma",
    )
    _save_view(view, "func_mean_view")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- atlas (cheap, always works) ---------------------------------- #
    try:
        atlas_img = fetch_cc200_atlas()
        figure_atlas_cc200(atlas_img)
        figure_atlas_view(atlas_img)
    except Exception as exc:  # noqa: BLE001
        print(f"      [fail] atlas section: {exc}")
        traceback.print_exc()

    # ---- one subject's actual scan (downloads ~50-150 MB) ------------- #
    try:
        func_img, sub_id, label = fetch_one_func()
        mean_img = figure_func_mean(func_img, sub_id, label)
        figure_func_view(mean_img, sub_id, label)
    except Exception as exc:  # noqa: BLE001
        print(f"      [fail] subject scan section: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Brain visualizations written to:")
    print(f"  {VIS_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
