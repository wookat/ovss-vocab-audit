# Pre-registration W12-K2: contamination-curve protocol — effective-vocabulary confound (frozen before any run)

Date frozen: 2026-08-01, before any contamination-curve run.

## Hypothesis (tenth-round ideator K2)
Benchmark mIoU of dense open-vocabulary segmenters depends strongly on
the number of absent (never-present-in-image) vocabulary entries, so
cross-benchmark comparisons (VOC-21 vs Context-60 vs ADE-150) partly
measure vocabulary composition rather than capability. A
contamination-curve protocol — mIoU as a function of injected absent
entries at fixed present classes — makes this confound measurable and
could change cross-benchmark conclusions.

## Frozen design
Base: VOC-21 plain, test-300, SCLIP and NACLIP.
Contamination pools, injected cumulatively at sizes n in
{0, 10, 25, 50, 100, 150}:
- pool R (random): class names sampled from ADE-150/ctx60 vocabularies
  minus VOC synonym/hypernym overlap (frozen seed 0);
- pool N (near): the existing dis_near candidates ranked by CLIP cosine
  to VOC classes (existing frozen artifacts).
Metrics per point: GT-present mIoU (present-class convention) and
all-class mIoU (fixed denominator; expected to fall mechanically).
Key measured quantity: GT-present mIoU degradation curve — real damage
beyond the convention term — and its pool dependence.

## Criteria (frozen)
- GO (protocol contribution): GT-present mIoU at n=150 drops by >= 3
  for the near pool on at least one model AND near-vs-random pools are
  separated by >= 2 at n=150 (composition matters beyond count) — then
  the contamination curve carries information a single-point benchmark
  hides, and the protocol section is warranted.
- NO-GO: GT-present drop < 1.5 at n=150 on both models for both pools —
  dense models are effectively insensitive to absent-entry injection in
  the present-class convention; the confound reduces to the known
  all-class convention term already covered by the BA decomposition.
- MIXED otherwise.

## Scope
One dataset (VOC), two models, single seed pools, test-300; positive
results require a Context-60 replication before any protocol paper
claim.
