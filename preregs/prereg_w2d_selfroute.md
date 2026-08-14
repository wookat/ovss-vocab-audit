# Preregistration W2d: SelfRoute go/no-go (frozen 2026-07-31, before any run)

## Question
Can cross-configuration consistency signals (optionally combined with DINO
self-supervised clustering boundary agreement) route SAM regions to inference
configurations better than the best single configuration — i.e. access part of
the +14.9 per-region oracle that margin/confidence/agreement signals (C6r2) and
a 50-image supervised ridge router (W1b) could not?

## Setup (frozen)
- Backbone ViT-B/16 openai, unified protocol (short 336 / win 224 / str 112 / scale 40).
- Configuration pool K=8: {vanilla,qq,kk,ident} x exit layers {11,12}
  (subset of the C6 pool covering both flavor and depth axes).
- SAM ViT-B automasks (points_per_side 16), same checkpoint as REVA.
- Datasets: VOC-21 images 300-499 (disjoint from dev-100 = 0-99) and
  Context-60 images 0-199. Plain vocabularies.
- DINO features: dino_vits16 (torch hub) final-layer patch features; region
  boundary agreement = mean cosine dissimilarity across the region boundary
  relative to region interior (higher = crisper visual region).

## Arms (all label-free at routing time; GT only for scoring)
1. best-single: best configuration of the pool selected on the SAME evaluation
   set (generous baseline, biased in its favor).
2. consist-argmax: for each region, route to the configuration whose region
   prediction agrees with the majority vote across the 8 configurations,
   tie-broken by highest mean region probability.
3. consist+dino: arm 2 score multiplied by the DINO boundary-agreement factor.
4. control margin-argmax: route by region margin (C6r2 signal, same pool) —
   mandated control so "changed supervision form" is the tested variable.
5. oracle: per-region best configuration by GT (ceiling re-measured on this pool).

## Kill criteria (frozen)
- K1: fraction of regions where the consistency pseudo-label matches the oracle
  configuration <= random baseline (1/8) + 10pp on both datasets -> kill.
- K2: consist-argmax (best of arms 2-3) mIoU < best-single + 2.0 on BOTH
  datasets -> kill (recovers <13% of oracle gap).
- Margin control performing as well as consistency arms => "supervision form"
  claim dies even if K2 passes.

## Disclosures
- Regions with no coverage fall back to the best-single prediction.
- The pool differs from C6 (8 vs 20 configs); oracle ceiling re-measured, not
  assumed equal to +14.9.
- Failure is reported as a negative result closing the "consistency signal
  family" alongside margin (C6r2) and few-label (W1b).
