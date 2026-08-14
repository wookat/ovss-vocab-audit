# W13-M1: pruning profit/loss ledger (bounded analysis, frozen 2026-08-01)
Question: why does J5's gain shrink VOC->ADE, and why does mispruning
1/3 of present classes not destroy the net gain?
Design: SCLIP + NACLIP x VOC-21 + ADE-150 plain, test-300, THRESH=0.2
(frozen J5 protocol unchanged). Pixel ledger over changed pixels:
- B (profit): dense-wrong -> pruned-correct pixels;
- A (loss): dense-correct pixels destroyed because their GT class was pruned;
- C: relabeled but still wrong.
Plus, for wrongly pruned present classes: their dense per-class IoU and
GT area share vs kept present classes (fault-tolerance hypothesis:
mispruned classes are small/low-IoU, so deleting them costs little).
Verdict rule (frozen): if the A:B structure differs >= 2x between VOC
and ADE, that identifies the shrinkage side (thinner profit vs thicker
loss) and is written into the J5 subsection; otherwise the shrinkage
stays an open note. Bounded 0.5-day analysis; no kill criteria.
Status: interpretation-only note; does not alter any frozen claim.
