# VisionCore agent guidance

## Exploratory analysis workflow

For new or poorly understood analyses, prefer a concrete, map-first progression:

1. Reduce the question to the smallest interpretable contrast.
2. Visualize the input or proposed mechanism before fitting broad summary models.
3. Inspect multiple raw activation maps across conditions and time.
4. Examine direct difference maps and detailed panels for interesting units.
5. Select example units using explicit, auditable roles and save the selection criteria and values to a table.
6. Include positive examples, dissociations, and negative or control examples.
7. Compute group and population summaries only after the example-level behavior is understood.

Treat intermediate stages as human-AI checkpoints. Present the artifacts, visible observations, surprises, and smallest useful next step, then wait for direction unless the user explicitly requests an autonomous end-to-end run.

Keep predicted outcomes labeled as hypotheses rather than assumptions. Preserve traceability from aggregate conclusions back to the underlying maps, examples, commands, configuration, and saved selection table.

Use the repo skill `map-first-analysis` for the detailed protocol when the task matches it.
