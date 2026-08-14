# Pre-registration W11-J5: detector-guided vocabulary pruning for dense CLIP (frozen before any run)

Date frozen: 2026-08-01, immediately after the out-of-prereg J4
observation (SCLIP plain 34.75 -> 40.80 under OWLv2 box-support pruning),
BEFORE any further run. The J4 plain cell is the motivating observation
and is NOT reused as evidence.

## Hypothesis
Image-conditioned vocabulary subsetting by an external detector's box
support removes absent-class competitors from the dense argmax and
improves dense CLIP segmentation on clean vocabularies — a third
training-free component candidate (orthogonal to VABS negatives and SAM
arbitration, different signal family from all dead presence gates: the
signal is an external heterogeneous detector, used to prune, not to gate
distractors).

## Frozen design
- Shield rule identical to J4 (OWLv2 base-ensemble, "a photo of a X",
  threshold 0.2, entry supported if >= 1 box; background always kept;
  unsupported entries removed from the argmax).
- Cells: {SCLIP, ClearCLIP, NACLIP} x {VOC-21 test-300, Context-60
  test-300, COCO-Object test-300} with plain vocabularies.
- Also: interaction with REVA on SCLIP/VOC (prune + VABS64 + SAM
  arbitration) to check composability.
- Controls: (a) trivial baseline — OWLv2+SAM standalone on the same
  vocab (archived); the claim is about improving the DENSE model, so the
  comparison of record is dense-vs-dense-pruned, but the standalone
  number must be disclosed alongside; (b) synonym harm — syn100_s0 on
  SCLIP/VOC with pruning (the detector may fail to support renamed
  classes and prune true classes away).

## Criteria (frozen)
- GO: mean improvement over the 9 dense cells >= +2 mIoU AND no cell
  harmed by more than 1 AND synonym-arm harm < 2 (pruning must not
  amplify naming fragility).
- NO-GO: mean improvement < +1 or any cell harmed > 3 or synonym-arm
  harm >= 4.
- MIXED: in between.

## Scope guards
Adds a second model at inference (cost disclosed); threshold not tuned;
single seed; test-300 subsets; VOC/COCO classes are in OWLv2's
detection-friendly distribution — Context-60 stuff classes are the hard
honest cell (boxes are a poor fit for stuff; expected failure mode is
pruning true stuff classes).
