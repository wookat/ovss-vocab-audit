# Pre-registration W13-L2: J5 defensive backfill — trivial top-k baseline + soft pruning (frozen before any run)

Date frozen: 2026-08-01, before any run.

## Question (reviewer-anticipated attack points)
(a) Is the J5 pruning gain just a trivial image-level class-filtering
effect that CLIP itself can provide (no detector needed)?
(b) Does soft pruning (down-weighting unsupported classes) mitigate the
ADE-150 gain shrinkage attributed to detector recall degradation?

## Frozen design
(a) Trivial baseline: CLIP image-level top-k filtering. Global image
embedding (same OpenCLIP B/16), rank vocabulary entries by image-text
cosine, keep top-k foreground classes + background, k set per image to
the number of classes OWLv2 pruning keeps (matched budget). Cells:
SCLIP + NACLIP x VOC plain + ADE-150 plain, test-300.
(b) Soft pruning: unsupported classes get logits scaled down by
lambda in {0.3, 0.5, 0.7} (on softmax probabilities: multiply prob by
lambda then renormalise; equivalently add log(lambda)). Same cells.

## Criteria (frozen)
(a) TRIVIAL: CLIP top-k reaches >= 80% of the OWLv2 pruning gain
(averaged over the four cells) -> the detector component collapses;
J5 subsection must be rewritten around image-level filtering.
NON-TRIVIAL: < 50%. MIXED between.
(b) SOFT-WINS: best lambda beats hard pruning on ADE by >= 1 mIoU while
losing <= 0.5 on VOC -> soft pruning enters the J5 subsection as the
recommended large-vocabulary form. SOFT-FLAT: no lambda improves ADE by
>= 0.5 -> hard pruning stays, recall degradation is not mitigated by
down-weighting. MIXED between.

## Scope
Two models, two datasets, single seed, test-300. Either outcome is
disclosed in the REVA J5 subsection.
