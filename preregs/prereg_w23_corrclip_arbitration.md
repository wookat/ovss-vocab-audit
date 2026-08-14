# Prereg W23: full REVA arbitration (additive) on official CorrCLIP, VOC-21 (frozen before runs)

Date frozen: 2026-08-03 (before any W23 arm is launched)

## Motivation
The REVA limitation currently reads: full REVA (VABS + SAM region
arbitration) is validated on one official release only (SC-CLIP, W16-E).
CorrCLIP is the natural second candidate BUT is itself already
region-arbitrated: its official postprocess does per-region mode voting
over its pre-generated SAM2 instance masks ("Map Correction"). W23 is
therefore a *redundancy test*: does our independent SAM (ViT-B automatic
masks, the same generator as W16-E) arbitration layer add anything on top
of a host that already votes over SAM2 regions?

## Protocol (frozen)
- Official CorrCLIP forward and postprocess unmodified; we consume the
  per-class probability map the repo itself stores in
  data_samples.seg_logit (post softmax/背景 max-fold, BEFORE its own map
  correction, as stored by the repo), pool it over our SAM ViT-B automatic
  masks (points_per_side=16, larger masks first, uncovered pixels keep the
  pixel prediction), re-apply the repo's own prob_thd identically --- the
  exact W16-E mechanics, sam_vit_b_01ec64 checkpoint.
- Comparison target: the official pipeline's own final prediction
  (pred_sem_seg, which includes the repo's map correction). This is the
  honest baseline: our layer must beat the shipped postprocess, not a
  stripped one.
- Dataset: VOC-21 full val (passed the W20 reproduction gate).
- Arms (2): plain+VABS64 (name file from W21), plain+rand64 (seed 0).
  Plain-only pixel references reused from W20/W21 logs.
- Single run per arm.

## Frozen predictions and criteria
- Primary criterion (same as W16-E): arbitration gain = ours - official
  postprocess, on the VABS arm. PASS if >= +1.0 mIoU.
- Honest frozen prediction: FAIL or marginal is the *expected* outcome ---
  the host already performs region mode voting over SAM2 masks, so a
  second region-pooling layer is plausibly redundant (gain in [-1, +1]).
  If it fails, the verdict is a scope boundary: "our arbitration layer
  adds value on pixel-level hosts (SC-CLIP +3.5) but is redundant on
  hosts that already arbitrate over regions" --- this SHARPENS the REVA
  mechanism claim (the gain comes from region evidence, however it is
  injected) rather than weakening it.
- Secondary (descriptive only): selection under arbitration
  (VABS+SAM vs rand+SAM), no threshold.

## Outcome handling
Either outcome is publishable as stated above; failures preserved. No
claim that full REVA "transfers to CorrCLIP" unless the +1.0 criterion
passes; if it fails, Limitations keeps "full REVA validated on one
official release" and gains the redundancy-boundary sentence.
