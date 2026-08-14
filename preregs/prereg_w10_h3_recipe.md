# Pre-registration W10 (H3): training-recipe natural experiment go/no-go (frozen before run)

Date frozen: 2026-08-01, before downloading any MM-Grounding-DINO weights.

## Question
W7b left the OWLv2-vs-others synonym-robustness contrast attributed only as
"consistent with a training-recipe effect". MM-Grounding-DINO offers a
same-architecture open checkpoint ladder differing only in training data
mixture. Does synonym robustness track the data recipe when architecture is
held fixed?

## Models (frozen; all MM-GDINO Swin-T, same config family)
Ladder tiers (as released by OpenMMLab, arXiv 2401.02361):
- T1: O365 (or the smallest released mixture)
- T2: O365 + GoldG
- T3: the largest released Swin-T mixture (O365+GoldG+GRIT+V3Det or "ALL")
If a tier's weights are not downloadable, substitute the nearest released
tier and disclose (as with the W7b GLIP->MDETR substitution).

## Harness (frozen)
Same box->SAM-B harness as F2/W7b (detector boxes -> SAM masks -> pixel
argmax by score), VOC-21 test-300, GT-present mIoU, detection threshold
identical across tiers. Vocabularies: plain and syn100_s0.

## Criteria (frozen)
- Infeasible kill: any tier's plain GT-present < 35.
- GO (recipe signal): synonym drop varies monotonically across the three
  tiers with total spread >= 5 mIoU.
- NO-GO: spread < 3 mIoU (recipe does not explain; attribution stays open
  and is recorded as such).
- MIXED: spread in [3,5) or non-monotone -> report observationally, no
  recipe claim.

## Scope guards
Observational even if GO (data mixture co-varies with query-format details
across tiers); detector-harness numbers not comparable to dense pipelines;
single seed, test-300.

## Cost
mmdetection env install ~0.5 day; 6 runs ~1 GPU day.
