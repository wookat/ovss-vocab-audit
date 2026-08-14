# Preregistration W5e: BA-decomposition protocol (frozen 2026-08-01)

Ideator-5 candidate E. Upgrade the qualitative "background absorption + metric
artifact" two-factor story into a quantitative pixel-flow decomposition of
perturbation damage, as a protocol extension of the Robust-mIoU benchmark.

## Method (frozen)
For each (method, vocab) run we store the full pixel confusion matrix
C[gt, pred] on VOC-21 test-300 and COCO-Stuff-171 val-300 (two-dataset
stability check; COCO-171 has no background class, so its BG-absorption term
is structurally zero and damage concentrates in steal/inter/artifact —
disclosed, amended before any run because ctx60 lacks a distractor vocab). For a perturbed vocabulary relative to the plain baseline,
decompose the per-class IoU loss into flow terms computed from C:
- BG-absorption: GT-class pixels predicted as background (or VABS negatives
  where present) beyond the plain-baseline rate.
- Distractor-steal: GT-class pixels predicted as injected distractor items
  (distractor axis only).
- Inter-class confusion: GT-class pixels predicted as other GT classes beyond
  baseline rate.
- Metric artifact: change in mIoU attributable purely to averaging over the
  enlarged class set (recompute mIoU restricted to GT classes vs all classes;
  the gap is the artifact term).
Methods: all 7 (existing pipeline). Vocabularies: plain, syn100_s0,
dis_near200.

## GO / KILL (frozen)
- GO: on the distractor axis, the sum of metric-artifact + BG/steal terms
  accounts for >= 50% of the all-class mIoU drop on >= 2 datasets, with the
  decomposition qualitatively stable (same dominant term) across >= 5 of 7
  methods.
- KILL (as standalone-paper material): if the BA-corrected method ranking is
  essentially unchanged (Spearman > 0.95 vs uncorrected on every axis), the
  decomposition merges into the benchmark appendix instead of standing alone.

## Disclosures
Offline analysis over new confusion-matrix runs under the frozen protocol; no
threshold tuning. Distinct from RENOVATE (name quality) and generic VLM
audits (no pixel-level flow attribution); no OVSS perturbation-loss
decomposition protocol found in the 5th ideation arXiv sweep.
