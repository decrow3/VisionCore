# Panel C Motion-Versus-Stabilized Representation Diagnostic

Question: does the V1 twin population represent the image-feature target
better with measured 1x motion than with the 0x stabilized counterfactual?

The clean oracle comparison is `known_eye - zero_static`. `known_eye`
uses the measured trajectory, so it removes latent eye-position
uncertainty from the representation question. `full_exact` keeps the
eye trace hidden and is included as the joint-decoder comparison.

At the 1x scale:

```text
0x stabilized feature cosine:        0.6678
1x motion, eye hidden feature cosine: 0.8721
1x motion, eye known feature cosine:  0.9358

oracle 1x gain over 0x:              0.2680
hidden-eye 1x gain over 0x:          0.2043
latent-eye penalty:                  0.0637
```

Interpretation: the oracle known-eye comparison supports the claim that
the moving 1x response carries more recoverable local image-feature
information than the stabilized 0x counterfactual. The smaller full-joint
gap shows how much of that representational advantage remains when eye
position is hidden.

Outputs:

- `panel_C_motion_vs_stabilized_representation.png`
- `panel_C_motion_vs_stabilized_representation.pdf`
- `panel_C_motion_vs_stabilized_representation_summary.csv`
- `panel_C_motion_vs_stabilized_representation_contrasts.csv`
