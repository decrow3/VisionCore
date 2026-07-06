# Figure 4D Linear-Gaussian Feature-Model Check

Minimal sanity check for the along/across contour result.

Inputs are the same matched-static panel-D response tables and the same
`pyramid_local_field` PCA feature target. The likelihood is replaced by a
trial-heldout ridge linear-Gaussian feature-to-response model.

Panel-matched contrast: `axis_edge_parallel - axis_edge_orthogonal`
in feature-recovery gain over the zero-eye baseline, using `-MSE`.
The summary figure also shows raw feature recovery and cosine contrasts,
because the linear-Gaussian zero baseline is a deliberately different
model from the original Poisson observer baseline.

Outputs:

- `linear_gaussian_panel_d_trials.csv`
- `linear_gaussian_panel_d_summary.csv`
- `linear_gaussian_panel_d_contrasts.csv`
- `linear_gaussian_panel_d_check.png`

Feature-space variance fraction: 0.328606
Ridge alpha: 10
Posterior temperature: 0.01
