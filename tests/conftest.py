"""Shared fixtures for the CS5491 NSC test suite."""

import pytest

from cs5491_nsc import TransformerLayerSpec

# The course evaluation convention fixes the initialization std at 0.02.
COURSE_INIT_STD = 0.02


@pytest.fixture
def init_std() -> float:
    return COURSE_INIT_STD


@pytest.fixture
def ascending_widths() -> list[TransformerLayerSpec]:
    """Two layers with narrow-then-wide FFN."""
    return [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
    ]


@pytest.fixture
def descending_widths() -> list[TransformerLayerSpec]:
    """The same widths as `ascending_widths`, FFN order reversed."""
    return [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    ]
