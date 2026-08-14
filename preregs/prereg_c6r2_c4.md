# Pre-registration: C6 round-2 label-free routing + C4 visual-side consistency adapter

Frozen 2026-07-30, before implementation. Unified protocol as before.

## C6 round 2: label-free per-region config routing (probe_route_lf.py)

Config pool (frozen, label-free selection): top-4 of the 20 configs by whole-output
label-free margin from headlens_e2_dev100.json (no GT involved). Regions: SAM ViT-B
pps=16 on the full image. Gate signals tested (all label-free, each an arm):
- G1: per-region mean top1-top2 margin of pooled class probabilities per config;
  route each region to the config with max region margin.
- G2: per-region max mean confidence (top1 prob).
- G3: cross-config agreement: route to the config whose region label agrees with the
  majority vote across the pool (ties -> G1).
Uncovered pixels: config with best global (whole-image) margin.

Kill / go criteria (frozen):
- K-C6r2a (go): best gate arm >= best single config of the FULL 20-config set + 5.0
  mIoU on dev-100 (i.e. >= 58.2 given best single 53.23).
- K-C6r2b: the routed result must also beat every pool member individually.
- Report the 4-config oracle for context (fraction of ceiling recovered).
- If all arms < +2: direction killed. Between +2 and +5: disclosed redesign allowed
  once (different gate signal), then final verdict.

## C4: vocabulary-perturbation consistency training of a visual dense adapter
(c4_train.py / c4_eval.py)

Motivation: the naming-robustness objective failed on the text side (LexRO); the
visual side is the only untested degree of freedom. Train a lightweight residual
conv adapter on frozen dense patch features so that predictions become invariant to
synonym perturbations of the vocabulary.

- Data: the existing 8,000-image ADE20K-train NACLIP feature cache
  (/media/dell/DATA/ovss/lexro_cache, feat only; teacher labels unused). No GT.
- Adapter: 1x1 conv 512->256, GELU, 3x3 conv 256->512, zero-init last layer,
  residual on the 14x14x512 patch grid, output re-normalized.
- Training vocabulary: ADE-150 canonical names; per-class WordNet variant pool with
  CLIP cos in [0.70, 0.95], names in any held-out eval synonym file banned
  (same machinery as LexRO).
- Loss per batch: L_cons = mean symmetric KL between class distributions
  (logit scale 40, subquery amax pooling) under V=4 random variant samplings;
  L_anchor = KL(adapted canonical dist || frozen canonical dist) [teacher=frozen];
  L_drift = mean squared feature change. Total = L_cons + 1.0*L_anchor + 0.1*L_drift.
  10 epochs, Adam 1e-4, batch 32 images.
- Collapse monitor: mean feature-change norm and prediction entropy each epoch;
  abort if entropy collapses to argmax-constant or feature change norm > 0.5.

Evaluation (NACLIP primary since the cache is NACLIP; SCLIP transfer reported as
secondary, disclosed): VOC-21 plain, official, and HELD-OUT synonym vocabularies
(same files as LexRO K3), dense pixel protocol, dev-excluded where applicable,
test-300 (offset 0..300).
Frozen baselines to be measured before training and recorded in RESULTS.md.

Kill / go criteria (frozen):
- K-C4a (go): held-out synonym mIoU (NACLIP, adapter) >= frozen baseline (no harm)
  AND plain mIoU >= frozen + 2.0.
- K-C4b: official vocab drop > 1.0 kills.
- If held-out synonym degrades like the text side did, the run is a negative result
  supporting the stronger claim "naming robustness is not learnable on either side
  in isolation" — preserved and reported, not hidden.
