# Pre-registration W6-F3: Vulnerability-prediction law (frozen before analysis)

Date frozen: 2026-08-01. Round-6 candidate F3 (round6-ideation).

## Hypothesis
A zero-evaluation, text-only geometry signal computed from a vocabulary's
class embeddings predicts the segmentation mIoU that vocabulary attains on a
given method, well enough to rank vocabularies without running segmentation.

## Signal (frozen)
For a vocabulary V with per-class pooled embeddings t_c (unit-norm, same
class_embeddings() pooling as the main pipeline):
- g1(V) = mean over classes of nearest-neighbour cosine margin
  (1 - max_{c' != c} cos(t_c, t_{c'})).
- g2(V) = mean cosine similarity of each perturbed name embedding to the
  plain-name embedding of the same class (semantic drift; = 1 for plain).
- Predictor: rank by the frozen combination z(g1) + z(g2) (z-scored across
  the vocabulary pool per dataset). No fitting on held-out targets.

## Evaluation (frozen)
Targets: existing archived benchmark runs (no new segmentation runs):
7 methods x vocab suite on VOC-21 full (w4a: plain, official, syn100_s0,
syn50_s0/s1/s2) and, as the second/third datasets, COCO-Object and
Context-60 (w3d: plain, syn100_s0, syn50_s0/s1/s2). Amended from COCO-171
before any analysis: the archived synonym suites live on cocoobj/ctx60;
COCO-171 has only 3 archived vocabularies (too few ranks). For each method,
Spearman between predicted rank
and actual mIoU rank across vocabularies; leave-one-method interpretation:
signal is text-only so it is constant across methods -- the test is whether
one signal ranks vocabularies for EVERY method.

## Criteria (frozen)
- GO: median per-method Spearman >= 0.7 on VOC AND >= 0.5 on both cocoobj
  and ctx60.
- PARTIAL: >= 0.7 on VOC only -> report as VOC-scoped observation.
- KILL: median < 0.5 on VOC.
Distractor-axis vocabularies are excluded from the ranking pool (their
all-class collapse is a metric-convention effect per Appendix badecomp, not
an accuracy loss); disclosed here, frozen before analysis.

## Cost
Text encoder forward passes only (~minutes). No GPU segmentation runs.
