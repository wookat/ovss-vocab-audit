# Pre-registration W7c: cross-lingual naming axis quick test (frozen before run)

Date frozen: 2026-08-01. Round-7 candidate G4. Question: does a translation
axis (same concepts, non-English names) produce a structured degradation
profile distinct from the synonym axis, or a floor collapse with no
discriminative power?

## Setup (frozen)
Vocabularies: VOC-21 class names translated to Chinese (zh) and Spanish
(es); fixed dictionary translations written before any run (single common
translation per class, no cherry-picking among variants). Background name
translated too.
Models: SCLIP (CLIP-surgery family) and OWLv2+SAM harness. VOC-21 test-300.
Metrics: GT-present mIoU vs plain English; compare against each model's
synonym drop (SCLIP -4.2 syn100 band; OWLv2 -7.1).

## Criteria (frozen)
- GO (structured axis): at least one model retains >= 50% of its plain
  mIoU on es or zh AND the cross-lingual drop pattern decouples from the
  synonym drop (rank of models differs across the two axes) -> promote to
  a fourth audit axis, run the full method matrix.
- NO-GO (floor collapse): all model x language cells fall below 10 mIoU ->
  no discriminative power; record one paragraph in limitations only.
- MIXED: es structured but zh collapsed (expected for English-only
  tokenizers) -> report es as the axis, zh as a boundary note.

## Cost
4 cells (2 models x 2 languages), < 0.5 GPU day.
