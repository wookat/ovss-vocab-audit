# Pre-registration W6-F2: cross-family audit transfer (grounding/detection family) (frozen before run)

Date frozen: 2026-08-01. Round-6 candidate F2. Question: is naming
fragility a CLIP-surgery artifact or does it persist under end-to-end
grounding training?

## Setup (frozen)
Models (fully open weights): Grounding DINO (IDEA-Research/grounding-dino-base
via transformers zero-shot detection) + SAM ViT-B box-prompted masks
(Grounded-SAM style semantic segmentation: per-class text query -> boxes ->
SAM masks -> per-pixel argmax by detection score). Optional second model
OWLv2 (google/owlv2-base-patch16-ensemble) same harness.
Dataset: VOC-21 test-300. Vocabularies: plain, syn100_s0, dis_near200.
Metrics: GT-present and all-class mIoU, same IoUMeter as the main pipeline;
background = pixels claimed by no detection.

## Criteria (frozen)
- GO (distinct-pattern): the grounding family shows a degradation structure
  differing from the CLIP-surgery family by >= 5 mIoU on at least one axis
  (synonym drop or distractor all-class drop), in either direction.
- NO-GO (same-pattern): both axes within 5 mIoU of the CLIP-family pattern
  -> fold into benchmark paper as a section, no standalone story.
- KILL (infeasible): harness cannot reach plain-vocab GT-present mIoU >= 35
  on VOC (implementation too weak to support audit claims).

## Cost
Pure inference; models < 1GB (GDINO-base ~0.9GB) + SAM-B; ~1 GPU day.
