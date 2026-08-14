# Preregistration W2c: VocabUQ go/no-go (frozen 2026-07-31, before any run)

## Question
Do region-level conformal class-prediction sets, with a nonconformity score
built from (a) region top-1 probability/margin and (b) prediction variability
under vocabulary perturbation (synonym-substituted vocabularies), give
(1) valid coverage that is stable across perturbation axes and datasets, and
(2) informative (small) sets — the go/no-go for the VocabUQ direction?

## Setup (frozen)
- SCLIP, ViT-B/16 openai, unified protocol; SAM ViT-B regions (pps 16).
- Calibration: 500 SAM regions sampled from COCO-Object val images 0-99
  (region label = GT majority class over the region; regions with <50% GT
  majority or majority=ignore are dropped before sampling).
- Test: 300 regions each from VOC-21 images 300-399 and Context-60 images
  0-99, same sampling rule, disjoint from all dev/test splits used for tuning.
- Vocabulary ensemble per dataset: plain + 3 synonym-substituted variants
  (seeds 0/1/2, frozen perturb.py rule, 50% substitution — 100% substitution
  is seed-invariant and would collapse the ensemble).
- Nonconformity score of region r for class c:
  s(r,c) = 1 - mean_v p_v(c|r) + lambda * std_v p_v(c|r), lambda = 1.0 (frozen).
- Split-conformal calibration at target coverage 90% (alpha = 0.1).

## Read-outs
- Empirical coverage on each test set overall and per perturbation variant.
- Mean prediction-set size (fraction of vocabulary).
- Abstention probe: fraction of person/tv regions (VOC) whose plain-vocabulary
  errors fall in sets that are large/unstable (would abstain), i.e. whether
  abstention "explains" the naming-engineering harm.

## Kill criteria (frozen)
- K1: mean set size > 50% of the vocabulary at 90% coverage on either test
  set -> uninformative, kill.
- K2: coverage deviates from 90% by >5pp across perturbation variants or
  datasets (exchangeability broken, no stratified fix attempted in go/no-go)
  -> kill.
- K3 (narrative, not kill): if <50% of person/tv naming-harm regions are
  flagged by large/unstable sets, the "abstention treats background
  absorption" story is downgraded.

## Disclosures
- Calibration uses ~500 labelled regions: the method is inference
  training-free but NOT annotation-free; this will be stated in any write-up.
- Cross-dataset calibration (COCO -> VOC/Context) intentionally stresses
  exchangeability; K2 measures exactly this.
- Prior art: conformal segmentation exists in medical/binary settings
  (arXiv:2511.15406, ConformalSAM 2507.15803); no claim to "first conformal
  segmentation", only to the OVSS vocabulary-uncertainty combination.
