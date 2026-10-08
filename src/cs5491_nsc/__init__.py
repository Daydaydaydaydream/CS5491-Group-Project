"""Scoring primitives for the CS5491 Neural Spectral Capacity study."""

from .nsc_utils import he_sigma, mp_density, psi_mp, xavier_sigma
from .student_score import (
    TransformerLayerSpec,
    additive_ablation,
    ffn_residual_load,
    layer_capacities,
    original_nsc_score,
    residual_context_score,
    score_transformer,
)

__all__ = [
    "TransformerLayerSpec",
    "additive_ablation",
    "ffn_residual_load",
    "he_sigma",
    "layer_capacities",
    "mp_density",
    "original_nsc_score",
    "psi_mp",
    "residual_context_score",
    "score_transformer",
    "xavier_sigma",
]
