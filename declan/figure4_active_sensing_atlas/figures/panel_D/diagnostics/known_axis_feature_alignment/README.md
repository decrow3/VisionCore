# Panel 4D Known-Axis Feature-Alignment Diagnostic

This diagnostic tests the direct along/across question: for each saved
axis-conditioned trajectory sample, the true candidate's rotated response
movie is treated as the observation, and the same known trajectory index is
used to score candidate images. This is not the hidden-eye joint decoder.

Primary contrast: `axis_edge_parallel - axis_edge_orthogonal` in
known-axis posterior feature cosine. The main confidence intervals
and sign-flip tests are clustered by trial; row-level trajectory-sample
uncertainty is retained in the contrast CSV for auditing.

Outputs:

- `panel_D_known_axis_feature_alignment.png`
- `panel_D_known_axis_feature_alignment_summary.csv`
- `panel_D_known_axis_feature_alignment_contrasts.csv`
- `panel_D_known_axis_feature_alignment_trials.csv`
