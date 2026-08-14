# W15-E: per-class harm predictor (frozen before analysis)

Trigger: REVA incremental re-review — last strong-accept gap: can we predict
WHICH classes REVA helps less / harms, rather than only reporting the observed
range? Hypothesis (frozen before computing any correlation): classes whose text
embedding is close to the selected VABS negative set are the ones REVA
under-serves — their pixels are absorbed by nearby negatives into background.

## Design
Offline analysis, no new segmentation runs. Per-class REVA delta = per-class IoU
of sam_reg_vabs (w15_fullrand_s0_*, full dev-excluded split) minus the plain
pixel baseline (w4a_voc21full_{sclip,naclip}_plain). Predictor = max CLIP text
cosine between the class's plain name and the 64 VABS negatives (same prompt
template as the pipeline, computed once). Background class excluded (it is the
target of the negatives). Statistic: Spearman rank correlation over the 20
foreground classes, per variant.

## Frozen interpretation
- Predictor GO if Spearman rho <= -0.40 in BOTH variants (closer negatives =>
  smaller/negative delta): paper adds the predictor with the rho values and a
  practitioner rule ("check max class-negative cosine before deploying").
- |rho| < 0.40 in either variant: reported as a null predictor (negative result
  preserved); no alternative predictor is tried post hoc in this prereg.
