"""
Named experiment configurations for ablation study.

Each experiment is a full 5-fold CV run with a specific set of hyperparameters.
Results saved to results/experiments/{name}/ — separate checkpoints, metrics, figures.

Usage:
    python run_experiments.py                        # run all experiments
    python run_experiments.py --exp baseline aug_full  # run specific ones
    python run_experiments.py --compare              # compare saved results
    python run_experiments.py --list                 # list all experiment names
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str
    description: str

    # Data
    harmonize: bool = False

    # Model architecture (applied to ALL three models)
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    latent_dim: int = 64
    dropout: float = 0.3

    # Training
    n_epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # "adam" | "adamw"
    patience: int = 20
    phase1_epochs: int = 30
    cls_weight_max: float = 5.0
    kl_warmup_epochs: int = 30

    # Augmentation
    augment: bool = False           # Gaussian noise on input
    noise_std: float = 0.02
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.0


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, ExperimentConfig] = {

    # ── Baseline ─────────────────────────────────────────────────────────────
    "baseline": ExperimentConfig(
        name="baseline",
        description="Adam, no augmentation, no harmonization",
        optimizer="adam",
        augment=False,
        use_mixup=False,
        label_smoothing=0.0,
        harmonize=False,
    ),

    # ── Optimizer ────────────────────────────────────────────────────────────
    "adamw": ExperimentConfig(
        name="adamw",
        description="AdamW instead of Adam (decoupled weight decay)",
        optimizer="adamw",
        augment=False,
        use_mixup=False,
        label_smoothing=0.0,
        harmonize=False,
    ),

    # ── Augmentation ─────────────────────────────────────────────────────────
    "aug_noise": ExperimentConfig(
        name="aug_noise",
        description="Gaussian noise on FC vectors (std=0.02)",
        optimizer="adamw",
        augment=True,
        noise_std=0.02,
        use_mixup=False,
        label_smoothing=0.0,
        harmonize=False,
    ),

    "aug_mixup": ExperimentConfig(
        name="aug_mixup",
        description="Mixup augmentation + label smoothing 0.1",
        optimizer="adamw",
        augment=False,
        use_mixup=True,
        mixup_alpha=0.2,
        label_smoothing=0.1,
        harmonize=False,
    ),

    "aug_full": ExperimentConfig(
        name="aug_full",
        description="Noise + Mixup + AdamW + label smoothing (best combo)",
        optimizer="adamw",
        augment=True,
        noise_std=0.02,
        use_mixup=True,
        mixup_alpha=0.2,
        label_smoothing=0.1,
        harmonize=False,
    ),

    # ── Harmonization ────────────────────────────────────────────────────────
    "combat": ExperimentConfig(
        name="combat",
        description="ComBat site harmonization (expected to hurt due to site-diagnosis confound)",
        optimizer="adamw",
        augment=True,
        use_mixup=True,
        label_smoothing=0.1,
        harmonize=True,
    ),

    # ── Latent dimension ─────────────────────────────────────────────────────
    "latent_32": ExperimentConfig(
        name="latent_32",
        description="Smaller latent space (32) — more regularization",
        optimizer="adamw",
        latent_dim=32,
        augment=True,
        use_mixup=True,
        label_smoothing=0.1,
        harmonize=False,
    ),

    "latent_128": ExperimentConfig(
        name="latent_128",
        description="Larger latent space (128) — more capacity",
        optimizer="adamw",
        latent_dim=128,
        augment=True,
        use_mixup=True,
        label_smoothing=0.1,
        harmonize=False,
    ),

    # ── Architecture depth ───────────────────────────────────────────────────
    "deep": ExperimentConfig(
        name="deep",
        description="Deeper encoder [512, 256, 128]",
        optimizer="adamw",
        hidden_dims=[512, 256, 128],
        latent_dim=64,
        augment=True,
        use_mixup=True,
        label_smoothing=0.1,
        harmonize=False,
    ),

    # ── Regularization ───────────────────────────────────────────────────────
    "high_dropout": ExperimentConfig(
        name="high_dropout",
        description="Higher dropout (0.4) for stronger regularization",
        optimizer="adamw",
        dropout=0.4,
        augment=True,
        use_mixup=True,
        label_smoothing=0.1,
        harmonize=False,
    ),
}
