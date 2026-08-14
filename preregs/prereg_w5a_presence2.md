# Preregistration W5a: Presence-Gated REVA v2 (frozen 2026-08-01)

Lineage disclosure: W4d killed a presence gate built on rank-thresholded
top-K SAM-region-pooled softmax scores (precision 0.24 @ recall 0.99). This is
NOT a revival of that frozen design: it tests a DIFFERENT signal family, per
the same convention used for the routing-signal chain (margin -> few-label ->
consistency). If this family also fails, presence gating is dead for the
project.

## New signal family (frozen)
For each vocabulary item c:
1. Negative-contrast margin m(c): for each SAM region r, margin(r,c) =
   pooled-sim(r, c) - max over VABS negatives n of pooled-sim(r, n), on RAW
   cosine similarities (not softmax, avoiding the mass-dilution failure mode
   of W4d). Presence score s1(c) = max_r margin(r,c).
2. Support concentration s2(c): fraction of the top-20 regions by
   pooled-sim(r,c) whose argmax class is c (winner consistency).
Combined score: s(c) = s1(c) standardized within-image (z-score over vocab)
+ s2(c). Gate: keep c iff z(s1) >= 0 OR s2 >= 0.3 (frozen; no tuning).
Background always kept. Re-argmax over surviving classes; abstention recorded
for gated-out classes.

## Evaluation (frozen)
Methods: ClearCLIP, NACLIP (same as W4d for comparability). VOC-21 test-300.
Vocabularies: official, plain, dis_near200.
- K1 (repair): distractor all-class mIoU >= 20 on both methods
  (baseline ~4; W4d gate left it at ~4).
- K2 (no harm): official and plain mIoU within -0.5 of ungated.
- K3 (gate quality): presence precision >= 0.6 at recall >= 0.9 (GT used for
  evaluation only).
KILL if K1 or K2 fails on either method. No threshold tuning after this file.
