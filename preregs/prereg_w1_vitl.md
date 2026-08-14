# Pre-registration W1a: backbone generalization to ViT-L/14 (frozen 2026-07-31)

Motivation: both papers currently rest on one backbone (OpenCLIP ViT-B/16 openai);
both simulated reviews flagged this. Question: do the audit's vocabulary effects and
REVA's gains replicate on ViT-L/14 (openai, quickgelu), same unified protocol
(short=336, window 224, stride 112, scale 40), VOC-21 test-300 (offset 0)?

Cells (run_eval.py --model ViT-L-14-quickgelu):
- methods: SCLIP, NACLIP
- vocabs: plain, official, syn100_s0, dis_near200 (plain+200 near distractors),
  plain_vabs64
Plus SAM arm (probe_d1sam --model ...): SCLIP, vabs64 vs randneg64, test-300.

Disclosed approximations: VABS negatives and distractor pools were selected with
ViT-B text embeddings (vocabularies are frozen text artifacts, reused verbatim);
NACLIP gaussian std kept at 5.0 (not retuned).

Expected replication criteria (descriptive, not kill):
- R1 naming: official - plain gap >= +8 mIoU per method.
- R2 synonym: syn100 below plain by >= 3.
- R3 distractor: plain+200near does NOT collapse (mIoU_GT-present drop < 10) and/or
  rises, replicating the background-sink direction.
- R4 VABS: plain_vabs64 - plain >= +8.
- R5 SAM pooling: sam_reg_vabs - pix_vabs >= +1; selection vabs - rand >= +2.
Failures are reported as scope limitations of the papers, not hidden.
