# Continuous Joint Along-Versus-Across Trace Diagnostic

This cache-only diagnostic asks whether the promoted strict no-start joint
estimator shows the same along-contour advantage as the older Figure 4D
axis-prior readout.

Scope note: the observed response family in this cache is empirical. The
`axis_edge_parallel` and `axis_edge_orthogonal` labels refer to the
image-conditioned trajectory prior/catalog family used by the observer,
not to two newly rendered observed movies.

At 1x for the promoted continuous joint estimator:

```text
along-contour feature cosine:  0.9407
across-contour feature cosine: 0.9366
along-contour image accuracy:  0.7031
across-contour image accuracy: 0.7031
```

Across all scales, the paired continuous-joint contrast is:

```text
feature cosine along - across:       +0.0011
feature gain-vs-zero along - across: +0.0011
```

Interpretation: this promoted continuous estimator does not reproduce a
clean along-contour advantage. In feature cosine, along is slightly
lower at 0.5x and slightly higher at 1x/2x, with confidence intervals
crossing zero. In hard image ID, along is better at 0.5x, tied at 1x,
and worse at 2x. Therefore the older 4D along-axis story should not be
automatically transferred to the strict continuous joint estimator
without this caveat.

Outputs:

- `continuous_joint_axis_trace_diagnostic.png`
- `continuous_joint_axis_trace_diagnostic_summary.csv`
- `continuous_joint_axis_trace_diagnostic_contrasts.csv`
- `continuous_joint_axis_trace_diagnostic_trials.csv`
