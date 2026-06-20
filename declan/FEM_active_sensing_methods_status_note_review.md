# FEM Active-Sensing Methods Note Review

Status: review of `FEM_active_sensing_methods_status_note_repo_audited.docx`
against the style of `inhomogenous stimuli writeup.pdf`.

## Short Verdict

The repo-audited docx is useful as a broad status ledger, but it is not yet in
the same genre as the inhomogeneous-stimuli methods note. The reference PDF is
a methodological argument: it starts from broken assumptions, defines the
estimator, derives what changes, validates on synthetic/real data, then states
the production implication. The current FEM note starts in that direction, but
then broadens into a survey of every analysis branch. That makes it helpful for
internal orientation and weaker as a methods explanation.

## What Already Works

- The opening claim boundary is right: distributional/structural language is
  strong; exact biological optimality and universal axis claims are not.
- The repeated `Motivation / Method / Current finding / Missing pieces` blocks
  are a good match to the reference style.
- The three-model active-sensing scaffold is the right organizing frame:
  aggregate, local pairing, and joint posterior solve different questions.
- The document keeps null/diagnostic branches visible instead of quietly hiding
  them.

## Main Style Gap

The reference PDF is narrow and cumulative. It teaches one estimator family by
making the reader carry a small set of variables through the whole note.

The current FEM note is broad and ledger-like. It switches among estimator
correction, recorded covariance, compact translation geometry, BackImage
feature decoding, local pairing, joint observer, behavior geometry, whitening,
Vernier, GLMs, chart-swap controls, and denoising. Each section is locally
reasonable, but the reader is not forced through one mathematical spine.

## Recommended Split

### 1. Manuscript-Facing Methods Note

Keep this focused on the core Figure 4/active-sensing pipeline:

```text
1. Reafferent retinal movie formulation
2. V1-twin response map and feature target
3. Aggregate FEM-distribution model
4. Local image-trace pairing model
5. Joint image-and-eye observer
6. Image-axis/behavior geometry bridge
7. Claim boundaries and current canonical run status
```

This document should read like the reference PDF: define notation, state the
estimator, explain why each control exists, then state what can and cannot be
claimed.

### 2. Internal Audit Ledger

Move the broader recorded-data, covariance-closure, whitening, Vernier, GLM,
chart-swap, and denoising branches into a companion ledger. They matter, but
they interrupt the methods story unless the reader already knows why each one
belongs.

## Concrete Rewrite Moves

1. Replace the long current-analysis ledger with a shorter "evidence map":

```text
Recorded anchor -> V1-twin retinal transform -> aggregate FEM utility ->
local pairing sensitivity -> latent eye-image observer -> behavior geometry.
```

2. Give the three active-sensing models a shared notation block:

```text
I: image/window
tau: eye trajectory
y = f_theta(I, tau): V1-twin response movie
phi(I): image feature target
D(y, phi): feature-decoding score
```

3. For each model, add a compact estimator contract:

```text
Aggregate: E_{I, tau ~ family} D(f_theta(I, tau), phi(I))
Local: D(f_theta(I, tau_actual), phi(I)) - E_{tau ~ matched} D(...)
Joint: log sum_tau p(y | I, tau) p(tau), compared to known-eye and zero-eye
```

4. Make the controls explicit as design constraints, not afterthought caveats:

```text
OU: confined drift control
Brownian: generic diffusion control
rotated: same path magnitude, changed image-relative direction
matched-unpaired: same trace pool, wrong image pairing
edge-parallel/orthogonal: image-geometry axis controls
matched-static distractors: remove trivial static-response wins
```

5. Move unresolved branches into one final "negative and diagnostic controls"
   section. Use it to bound claims, not to expand the main method.

6. Tie the doc to the canonical production run state:

```text
Current two-readout target:
  aggregate/ensemble: pyramid_local_field k16 temporal_pca
  local sensitivity: pyramid_local_field k16 delta_mean

Canonical run status:
  model-selection adjudication remains provisional until joint rel0.25 closes;
  production k16 reruns should use guarded canonical wrappers.
```

## Biggest Content Caveat

Several numeric examples in the docx are marked as prior-analysis or
not-currently repo-auditable. That is appropriate for a status note, but a
methods note should either remove those exact values or put them in an
"historical motivation" box. The reference PDF can quote numbers confidently
because it is built around the estimator being validated; the FEM note should
avoid mixing old illustrative values with canonical production values.

## Suggested Next Artifact

Create a source markdown file for the docx, rather than editing only the binary
document:

```text
declan/FEM_active_sensing_methods_status_note_repo_audited_source.md
```

Then render docx/PDF from that source with Pandoc. This will make future LLM
edits, code review, and provenance diffs much cleaner.

