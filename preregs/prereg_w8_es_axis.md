# Pre-registration W8: Spanish naming axis, full method matrix (frozen before run)

Date frozen: 2026-08-01, after the W7c MIXED verdict (es structured / zh
collapsed) and before any further runs.

## Question
Is the cross-lingual (Spanish) degradation profile stable and decoupled
from the synonym axis across the full audited method set, so that it can
enter the audit paper as a fourth perturbation axis?

## Setup (frozen)
- Vocabulary: the frozen voc21_es.json from W7c (no re-translation).
- Dense CLIP-family: MaskCLIP, SCLIP, ClearCLIP, NACLIP + style-reimpl
  ProxyCLIP, LPOSS, SC-CLIP — same unified protocol (336/224/112, scale 40),
  VOC test-300.
- Detectors: OWLv2 (done in W7c, 57.3) and Grounding DINO via the box->SAM
  harness, VOC test-300.
- Metric: GT-present mIoU; retention = es / plain; compare per-method
  retention rank against synonym-axis retention rank (Spearman).

## Criteria (frozen)
- PROMOTE (fourth axis): >= 6 of 9 models produce a valid es run AND the
  es-retention ranking is decoupled from the synonym ranking (Spearman
  < 0.6) OR shows a stable family split (detector vs dense) as in W7c ->
  write the cross-lingual subsection into the audit paper.
- DEMOTE (limitations note): es retention is uniformly low (< 30% for all
  dense methods) with no rank structure -> one-paragraph boundary note
  only.
- Disclose per-model tokenizer/coverage issues (as with zh truncation) in
  either case.

## Cost
~9 runs, <= 1 GPU day.
