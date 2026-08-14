# Pre-registration: LexRO v1 (frozen 2026-07-31, before any training run)

## Claim under test
A text-side residual adapter + learned background queries, trained with (i) a
naming-invariance consistency loss, (ii) REVA pseudo-label self-distillation and
(iii) an anchoring/anti-collapse regulariser, on UNLABELLED images, makes frozen
dense-CLIP OVSS robust to inference-vocabulary choice at zero inference cost.

## Training setup (frozen)
- Data: 8000 ADE20K TRAIN images (seed 0 fixed subset of images/training; GT never
  read). Disjoint from every eval set (VOC val, COCO val2017, ADE val, PC-459 val).
- Teacher: REVA = NACLIP dense (unified protocol, 224 centre crop of the 336
  short-side image) + VABS-64 negatives for the COCO-171 plain vocabulary +
  SAM ViT-B automask region pooling. Cached offline (labels + confidence per patch).
- Training vocabulary: COCO-171 plain names. Variant pool: WordNet synonyms with
  CLIP cos in [0.70, 0.95] (perturb.py rule), MINUS any name occurring in any
  voc21_syn*/coco171_syn* eval vocab file (train/eval variant disjointness).
- Trainables: 2-layer residual MLP on text embeddings (512-256-512, zero-init out)
  + 16 background query vectors initialised from VABS facility-location picks.
  Vision tower, text tower fully frozen.
- Losses: L = L_distill (conf-weighted CE to teacher patch labels)
  + a*L_inv (symmetric KL between patch logits under two independently sampled
  naming-variant vocabularies) + b*L_anchor (1 - cos(adapted, frozen))
  + c*L_sep (hinge on increase of max off-diagonal class-class cosine vs frozen).
  Hyperparameters tuned ONLY on VOC dev-100 (images [300,400)).

## Frozen kill criteria (VOC clean split = 1349 dev-excluded images unless noted)
- K1 (amortisation): plain-vocab pixel-level mIoU gain of LexRO over frozen plain
  baseline must reach >= 70% of REVA's training-free gain on >= 2 of the 4 base
  methods. Else: amortisation failed -> kill or fall back to C2 discussion.
- K2 (official safety): official-vocabulary mIoU drop > 1.0 on any base method
  -> anchoring failed -> kill or redesign (one redesign round allowed, disclosed).
- K3 (name invariance, the novelty claim): on HELD-OUT synonym vocabularies
  (voc21_syn100_s0/s1/s2, never used in training), the mIoU spread (max-min over
  seeds) and the mean drop vs plain must both shrink by >= 40% relative to frozen
  CLIP on >= 2 base methods. Else the name-invariance claim is unsupported.
- K4 (transfer): gains must not be VOC-only: COCO-Object plain-vocab gain >= +2
  and no regression worse than -1 on ADE-150/PC-459 GT-present mIoU. Else it is
  base-class overfitting.
- K5 (per-class safety): person and tvmonitor IoU vs frozen plain baseline must
  not drop by > 3 on >= 2 base methods (the audit failure mode).
- Embedding effective rank monitored each epoch; collapse (rank < 50% of frozen)
  aborts the run regardless of mIoU.

## Honesty constraints
- Teacher ceiling: any claim that LexRO exceeds REVA must come with the L_inv
  ablation showing the excess disappears without it.
- All vocabularies frozen in perturbed_vocabs/ before evaluation.
- Report GPU-hours and peak memory (single 3090).
