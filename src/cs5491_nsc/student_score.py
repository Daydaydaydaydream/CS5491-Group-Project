"""Course-project scores built on the upstream NSC MP utility."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from .nsc_utils import psi_mp


@dataclass(frozen=True)
class TransformerLayerSpec:
    """The dimensions needed by the course's dense decoder score."""

    hidden_width: int
    heads: int
    ffn_width: int


def ffn_residual_load(
    hidden_width: int, ffn_width: int, init_std: float = 0.02
) -> float:
    """Estimate the output/input variance ratio for W2 ReLU(W1 x).

    Assumes independent Gaussian matrices, isotropic inputs, and no normalization
    or learned scaling. This proxy is a project hypothesis, not part of original NSC.
    """
    if hidden_width <= 0 or ffn_width <= 0:
        raise ValueError("hidden_width and ffn_width must be positive")
    if not isfinite(init_std) or init_std < 0:
        raise ValueError("init_std must be finite and non-negative")
    return 0.5 * hidden_width * ffn_width * init_std**4


def layer_capacities(
    layer: TransformerLayerSpec, init_std: float = 0.02
) -> tuple[float, float, float]:
    """Return (attention capacity, FFN capacity, FFN residual load)."""
    d, heads, ffn = layer.hidden_width, layer.heads, layer.ffn_width
    if d <= 0 or heads <= 0 or ffn <= 0:
        raise ValueError("layer dimensions must be positive")
    if d % heads:
        raise ValueError("hidden_width must be divisible by heads")
    if not isfinite(init_std) or init_std < 0:
        raise ValueError("init_std must be finite and non-negative")

    head_width = d // heads
    attention = 3 * heads * psi_mp(d, head_width, init_std) + psi_mp(d, d, init_std)
    ffn_capacity = 2 * psi_mp(d, ffn, init_std)
    load = ffn_residual_load(d, ffn, init_std)
    return attention, ffn_capacity, load


def residual_context_score(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
    alpha: float = 1.0,
) -> float:
    """Score layers with an FFN weight based on preceding residual load.

    For layer l, the score is A_l + F_l / (1 + alpha * sum_{j<l} q_j).
    ``alpha=0`` is the additive ablation and matches the original NSC sum under
    the course's selected matrices and fixed initialization convention.
    """
    n = len(attention_capacities)
    if len(ffn_capacities) != n or len(ffn_residual_loads) != n:
        raise ValueError("all per-layer sequences must have the same length")
    if not isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    total = 0.0
    prior_load = 0.0
    for attention, ffn, load in zip(
        attention_capacities, ffn_capacities, ffn_residual_loads
    ):
        if not all(isfinite(value) for value in (attention, ffn, load)):
            raise ValueError("capacities and residual loads must be finite")
        if attention < 0 or ffn < 0 or load < 0:
            raise ValueError("capacities and residual loads must be non-negative")
        total += attention + ffn / (1.0 + alpha * prior_load)
        prior_load += load
    return total


def additive_ablation(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
) -> float:
    """Return the additive NSC baseline over the selected matrices."""
    return residual_context_score(
        attention_capacities, ffn_capacities, ffn_residual_loads, alpha=0.0
    )


def score_transformer(
    layers: Sequence[TransformerLayerSpec],
    init_std: float = 0.02,
    alpha: float = 1.0,
) -> float:
    """Compute the residual-context score from a sequence of layer specs."""
    capacities = [layer_capacities(layer, init_std) for layer in layers]
    attention, ffn, loads = zip(*capacities) if capacities else ((), (), ())
    return residual_context_score(attention, ffn, loads, alpha=alpha)


def original_nsc_score(
    layers: Sequence[TransformerLayerSpec], init_std: float = 0.02
) -> float:
    """Compute the course's original additive NSC over selected projections."""
    return score_transformer(layers, init_std=init_std, alpha=0.0)
