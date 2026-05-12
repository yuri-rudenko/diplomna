"""
Training schedules: KL annealing and two-phase cls_weight ramp.
"""

from __future__ import annotations


def beta_schedule(epoch: int, warmup_epochs: int = 30, beta_max: float = 1.0) -> float:
    """
    Linear KL annealing: beta grows from 0 to beta_max over warmup_epochs.
    Prevents posterior collapse in early training.
    """
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


def cls_weight_schedule(
    epoch: int,
    phase1_epochs: int = 30,
    cls_weight_max: float = 5.0,
    warmup_epochs: int = 10,
) -> float:
    """
    Phase 1 (epoch < phase1_epochs): cls_weight = 0
    Phase 2 warmup (phase1 → phase1+warmup): linear ramp 0 → cls_weight_max
    Phase 2 stable (epoch >= phase1+warmup): cls_weight_max

    Gradual ramp prevents the loss spike at phase transition.
    """
    if epoch < phase1_epochs:
        return 0.0
    ramp = min(1.0, (epoch - phase1_epochs) / max(warmup_epochs, 1))
    return cls_weight_max * ramp
