# Pre-registration W12-K3: GT-presence oracle upper bound for the distractor axis (frozen before any run)

Date frozen: 2026-08-01, before any oracle run.

## Question
Four signal families failed to repair the distractor collapse. Is
presence information itself sufficient — i.e. if an oracle told us which
vocabulary entries are present in the image, would pruning to them
restore plain-level performance? This quantifies the value of perfect
presence estimation and decides whether the problem should be restated
from "vocabulary filtering" to "presence estimation".

## Frozen design
VOC test-300, dis_near200 vocabulary (21+200), three dense models
(SCLIP, ClearCLIP, NACLIP). Oracle rule: for each image, keep background
+ the GT-present classes only (from the image's GT mask); prune all
other vocabulary entries from the argmax (same mechanism as J5, support
set replaced by the GT oracle). Report all-class and GT-present mIoU vs
the plain 21-class baseline of each model.

## Criteria (frozen)
- GO (restatement supported): oracle recovers >= 70% of the plain
  baseline all-class mIoU on all three models.
- NO-GO: recovery < 40% on any model — collapse has roots beyond
  vocabulary contamination; rewrite mechanism claims instead.
- MIXED: between.

## Notes
Oracle uses GT presence only (not GT masks for assignment); this is an
upper bound, never a deployable method; single seed; test-300.
