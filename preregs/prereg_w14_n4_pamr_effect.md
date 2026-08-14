# W14-N4: do audit effect sizes survive publication-grade post-processing? (frozen 2026-08-02, before any run)

Trigger: rebuttal drill A1 — all headline effect sizes are measured in a
minimal pipeline without PAMR/multi-scale; a hostile reviewer argues the
magnitudes may not extrapolate to published operating points (spatial
smoothing could absorb naming noise).

Design: SCLIP + NACLIP, VOC-21 test-300, frozen protocol + PAMR
refinement (same PAMR settings as the existing NACLIP anchor row,
applied to pixel probabilities before argmax). Vocabulary conditions:
official, plain, syn100_s0. Six PAMR cells; the three no-PAMR cells
already exist in archived runs.

Metrics: GT-present mIoU; effect sizes NEG = official - plain and
syn-drop = plain - syn100.

Frozen verdicts (per effect, averaged over the 2 models):
- SURVIVES: PAMR effect size within +/-30% of the no-PAMR effect size
  for both NEG and syn-drop -> report as "publication-grade
  post-processing does not materially change the audited magnitudes".
- ATTENUATED: an effect shrinks by more than 30% -> report honestly as
  "post-processing partially mitigates; direction unchanged" (or, if
  direction reverses, as a scope limitation on the audit's magnitudes).
- AMPLIFIED: grows by more than 30% -> report as such.

Notes: PAMR is refinement on our own pipeline, not the full published
stack (no multi-scale, no official repo run) — this bounds the
post-processing axis specifically, and the disclosure will say so.
Single seed, test-300.
