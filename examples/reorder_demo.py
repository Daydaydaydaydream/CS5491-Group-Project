"""Controlled width-reordering example from the course topic brief."""

from cs5491_nsc import TransformerLayerSpec, original_nsc_score, score_transformer

if __name__ == "__main__":
    forward = [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
    ]
    reversed_widths = [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    ]

    print("Original additive NSC:")
    print(f"  [256, 512]: {original_nsc_score(forward):.6f}")
    print(f"  [512, 256]: {original_nsc_score(reversed_widths):.6f}")
    print("Residual-context score (alpha=1):")
    print(f"  [256, 512]: {score_transformer(forward, alpha=1.0):.6f}")
    print(f"  [512, 256]: {score_transformer(reversed_widths, alpha=1.0):.6f}")
