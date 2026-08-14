# Pre-registration W11-J4: cross-model box-evidence shield for the distractor axis (frozen before any run)

Date frozen: 2026-08-01, after J2/J3 verdicts, before any shield run.

## Question
The distractor collapse (all-class mIoU ~4 under +200 near-distractors) is
REVA's only unrepaired axis. All CLIP-internal presence signals are dead
(two families, W4d/W5a). Untested family: evidence from an external,
architecturally heterogeneous detector. W7 showed detectors barely steal
foreground under distractors (4.5-8.5% vs 29-43%). Can OWLv2 box evidence
shield a dense CLIP method's distractor predictions?

## Frozen design
- Dense base: SCLIP, VOC test-300, vocab = voc21_dis_near200 (21+200).
- Shield: run OWLv2 (per-class queries, standard threshold 0.1 as in the
  crossfam probe) on the same 221-name vocabulary. A vocabulary entry is
  "box-supported" if it has at least one detection above threshold in the
  image. Dense pixels assigned to a NON-supported distractor class are
  reassigned to the argmax over the supported + VOC-21 base subset...
  no: to the argmax over box-supported entries plus background. The
  21 base classes are NOT given a free pass: they are subject to the same
  support test (deployment-honest; the shield does not know which entries
  are distractors).
- Metrics: all-class mIoU (primary), GT-present mIoU, plain-vocabulary
  harm (same shield applied to the 21-class plain vocab).
- Trivial-baseline control: OWLv2+SAM standalone segmentation on the same
  221 vocab (its all-class mIoU is already archived / re-run if needed).

## Criteria (frozen)
- GO: shielded all-class mIoU >= 25 (from ~4) AND plain harm < 1 mIoU.
- KILL-trivial: if OWLv2+SAM standalone achieves >= 80% of the shielded
  all-class gain, the contribution is the detector, not the shield ->
  fold into REVA as related-work note only.
- NO-GO: shielded all-class < 15 or plain harm >= 2.
- MIXED: in between; a follow-up cross-dataset check (ctx60) is required
  before any paper claim if GO.

## Scope guards
Adds a second model at inference (disclosed cost); box threshold fixed at
the archived crossfam value, not tuned; single seed; test-300.
