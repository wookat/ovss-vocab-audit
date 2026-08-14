# Pre-registration W13-L4: detector granularity — does OWLv2 presence recall degrade with label granularity, and does it explain the ADE pruning-gain shrinkage? (frozen before any run)

Date frozen: 2026-08-01, before any per-class recall computation.

## Hypothesis
The J5 pruning gain shrinks on ADE-150 (+2.5..2.7 vs +6..9 on VOC)
because OWLv2's per-class presence recall degrades on finer-grained /
rarer labels, so more GT-present classes get wrongly pruned.

## Frozen design
ADE-150 test-300, OWLv2 threshold 0.2 (frozen J5 setting), reusing the
K4/L2 detection outputs where cached, else recomputed.
Per class c (foreground): presence recall = fraction of images
containing c where OWLv2 fires >= 1 box for c; wrong-prune rate =
fraction where c is GT-present but pruned.
Granularity measure: WordNet min hypernym depth of the class's first
lemma (frozen; classes with no synset are excluded and counted).
Frequency control: log pixel frequency of c in the ADE train split
convention (we use the test-300 GT pixel counts as proxy; frozen).
Analyses:
1. Spearman(depth, presence recall) across classes;
2. Spearman(presence recall, per-class J5 gain (pruned IoU - dense
   IoU) on SCLIP);
3. partial Spearman of (1) controlling for log frequency.

## Criteria (frozen)
- GO (granularity account): rho(depth, recall) <= -0.3 AND
  rho(recall, gain) >= 0.4 AND the depth-recall relation survives the
  frequency control (partial rho <= -0.2).
- NO-GO: |rho(depth, recall)| < 0.15 OR frequency control absorbs it
  (partial |rho| < 0.1) -> do not claim a granularity law; the
  shrinkage stays attributed generically to detector recall.
- MIXED otherwise.

## Scope
One detector, one dataset, one threshold, observational. Regardless of
verdict this feeds one boundary sentence in the REVA J5 subsection, not
a standalone paper.
