# Pre-registration W6-F1: RECAL -- transductive logit debiasing for the synonym axis (frozen before run)

Date frozen: 2026-08-01. Round-6 candidate F1 (round6-ideation). Target: the
only remaining un-repaired GENUINE accuracy loss (synonym axis = pure
inter-class confusion, zero convention effect; Appendix badecomp).

## Mechanism claim
Synonym substitution shifts per-name activation scales, producing systematic
inter-class flow. A prediction-space, name-conditioned additive logit bias
b_c estimated transductively from the unlabeled evaluation stream can undo
part of that flow. No embedding modification (text-side dead band), no
training, no labels.

## Method (frozen, minimal version)
1. Pass 1 over the evaluation images: accumulate the predicted-class pixel
   mass distribution p_hat(c) under the perturbed vocabulary (argmax of the
   standard pipeline logits).
2. Reference prior p_ref(c): the predicted-class mass distribution obtained
   from the SAME images and method under the plain-template mean-synonym
   ensemble is NOT used (would be prompt ensembling). Instead p_ref is the
   flow-balance prior: p_ref(c) proportional to p_hat(c)^alpha with
   alpha = 0.5 (mass-flattening; frozen), renormalized. Background excluded
   from re-balancing (its mass is scene-dependent, not name-dependent).
3. Bias: b_c = tau_b * (log p_ref(c) - log p_hat(c)), tau_b = 1/logit_scale
   spelled out: b applied on cosine-similarity scale as
   b_c / 40.0. Iterate pass-1/pass-2 EM style for 3 iterations (frozen).
4. Pass 2: logits_c - shifted by b_c, re-argmax. Optionally composed with
   REVA SAM arbitration afterwards (separate arm).

## Experiment (frozen)
Methods: SCLIP, NACLIP. Dataset: VOC-21 test-300. Vocabularies: plain,
syn100_s0, syn100_s1, syn100_s2 (synonym axis only). Arms: baseline pixel,
+RECAL. Report per-cell mIoU (GT-present and all-class identical here; no
vocab expansion).

## Criteria (frozen)
- GO: mean recovery of the synonym-axis drop (relative to each method's own
  plain run) >= 40% across the 6 synonym cells AND plain-vocab harm < 0.5
  mIoU (RECAL applied to plain must not hurt).
- KILL: mean recovery < 20%, or plain harm >= 1.0, or recovery on only one
  method (< 10% on the other).
- Disclosed risk: transductive setting (statistics from the test stream,
  no labels). If GO, a single-image online variant must be evaluated before
  any paper claim; batch-only efficacy will be labeled transductive.

## Cost
Pure inference + cheap EM: ~1 GPU day for the go/no-go grid.
