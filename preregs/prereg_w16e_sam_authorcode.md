# W16-E: SAM arbitration on official SC-CLIP logits (frozen before runs)

Trigger: W16-C/D anchor VABS only; reviewers list "SAM arbitration not
validated on any official release" as REVA's remaining external-validity gap.

## Design
Official SC-CLIP forward is left unmodified through its postprocessed
probability map (the per-class softmax + alias-max output it already stores in
data_samples.seg_logits); our SAM arbitration is applied strictly as an
ADDITIVE post-processing layer on top of those probabilities: SAM ViT-B
(sam_vit_b_01ec64.pth, points_per_side=16, the frozen REVA config) automatic
masks on the original image, larger-masks-first region map, region-mean
pooling of the official probabilities, uncovered pixels keep the official
pixel prediction, then the repo's own prob_thd=0.15 background rule is applied
identically to both arms. Full VOC val 1449, cfg_voc21.

Arms (all on official SC-CLIP logits):
1. pix_plain      = plain names, no SAM (must equal W16-C's 42.97 -- regression);
2. pix_vabs       = plain+VABS64, no SAM (must equal W16-C's 59.62 -- regression);
3. sam_plain      = plain names + SAM arbitration;
4. sam_vabs       = plain+VABS64 + SAM arbitration (= full REVA on official logits);
5. sam_rand       = plain+rand64 + SAM arbitration (selection control under SAM).

## Frozen interpretation
- SAM arbitration transfers if (4) - (2) >= +1.0 mIoU (in-stack SAM adds
  roughly +1-3 over pixel VABS on VOC).
- Full-REVA claim upgrade allowed only if BOTH (4)-(2) >= +1.0 and
  (4)-(5) > 0; otherwise the paper keeps "VABS-only anchored" for the gain
  and reports the SAM cell(s) as-is.
- If regression arms (1)/(2) deviate from W16-C by > 0.3 mIoU, debug harness
  before interpreting; no knob tuning.
- SC-CLIP's own PAMR stays exactly as its config ships it in all arms.
