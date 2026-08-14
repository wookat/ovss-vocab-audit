# Prereg W24: Structured Negative Basis (SNB) --- background subspace geometry + angle-based applicability criterion (frozen before runs)

Date frozen: 2026-08-04 (before any W24 computation is launched)

## Motivation
Owner directive: raise REVA's method-level novelty beyond the
combination-plugin ceiling. Ideator round 6 + intelligence collision scan
(2026-08-03) converged on the cleanest verified gap: background modelled
as a low-dimensional *subspace of the text embedding space* (continuous
basis) rather than a discrete negative-word set, plus a deployment-time
applicability criterion computed from vocabulary--subspace principal
angles. Closest prior art verified and to be cited/contrasted: CC
(2407.05061, discrete query-conditioned contrastive words), DiSa
(2601.20064, visual-side fg/bg decoupling), CASS (2411.17150, per-class
scalar reweighting). No direct prior for text-side background basis or
angle-based applicability (intel verdict SAFE).

## Method under test (frozen)
1. Negative bank: the frozen VABS candidate pool plus WordNet common
   nouns (~5-10k words), encoded once with the host's own CLIP text
   encoder + standard templates.
2. Background basis B in R^{d x r}: top-r right singular vectors of the
   mean-centred bank embedding matrix (plain SVD, training-free);
   r in {8, 16, 32, 64} ablation, primary r frozen at 32.
3. SNB background score per patch: norm of projection of the patch
   embedding onto span(B) (scaled to logit range by matching the mean of
   the replaced background channel), replacing the max-over-negative-words
   channel. Everything else in the host pipeline unchanged.
4. Applicability angle theta(V): mean principal angle between span of the
   input vocabulary embeddings and span(B). Hypothesis: small theta
   (vocabulary close to background subspace) <=> low NEG headroom <=>
   VABS/SNB harmful; large theta <=> recoverable gap.

## Experiments and frozen criteria
- H2 first (fastest falsification, near-zero compute): compute theta(V)
  for >= 12 existing configurations with known oracle negative-expansion
  gains (hosts x datasets from W16/W17/W21 official-code runs + in-stack
  SCLIP/NACLIP cells; gains already archived in RESULTS). Criterion:
  Spearman(theta, oracle VABS-minus-plain gain) >= 0.7 -> PASS;
  < 0.4 or unstable sign under leave-one-out -> FALSIFIED (then the
  angle criterion is dropped; the basis scoring H1 may still proceed).
  NOTE the honest prior: the per-class text-side predictor was NULL
  (W15-E); this is a different granularity (whole-vocabulary,
  subspace-level, predicting dataset-level applicability), but failure is
  a live possibility and will be reported either way.
- H1: SNB projection scoring vs frozen VABS-64 on SCLIP + NACLIP, VOC-21
  test-300 first, then full val if directionally positive. Criterion:
  mIoU(SNB) >= mIoU(VABS64) - 0.3 with reduced across-seed variance
  (no random word sampling exists in SNB by construction), and
  SNB > plain by >= +3.0 where VABS passes the same bar. FALSIFIED if
  SNB < VABS64 - 1.0.
- H3 (only if H2 passes): theta-thresholded switch achieves do-no-harm
  (loss <= 0.2 mIoU) on known harmful configs (CorrCLIP COCO NEG +1.4,
  Trident Context-60 +0.2, in-stack ADE) while retaining >= 90% of the
  oracle gain on large-NEG configs.

## Scope and outcome handling
Single seed unless stated; bank composition ablation (WordNet vs frozen
VABS pool) is descriptive. All outcomes reported; failures preserved. No
claim of a validated predictor unless H2+H3 both pass; H1 alone upgrades
the mechanism story (word-set -> subspace) but not the applicability
claim. Collision guard: cite and contrast CC/DiSa/CASS; do not use
"pruning/filtering" language (ActiveSAM/FreeCP territory).
