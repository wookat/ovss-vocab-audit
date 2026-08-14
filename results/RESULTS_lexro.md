# LexRO verdict (prereg_lexro_v1.md) — KILLED, negative result retained

All runs: 3090, teacher cache 8000 ADE train imgs (NACLIP + COCO171 VABS-64 + SAM,
~70 min), each training run < 15 min. Evals: VOC dev-100 (offset 300), SCLIP/NACLIP.

## Runs
- run1 (adapter+16 bgq, distill+inv+anchor+sep): plain sclip 30.5→39.9, naclip
  32.9→32.7. Gain almost entirely from bg queries (bg-only 40.5; adapter-only 30.7).
- run2 (64 bgq, lr 3e-4, w_anchor 0.1, 20 ep): sclip 38.7, naclip 35.3.
- run3 (disclosed redesign 1: per-batch vocab subsampling so bg learns to absorb
  out-of-vocab): WORSE (sclip 31.6, naclip 20.6).
- v2 (disclosed redesign 2: pure text-side name normalization, variants→canonical
  cosine regression): uniformly harmful (plain 30.5→22.8, syn100 26.9→20.7,
  vabs 52.8→48.5, official →48.0).

## Kill criteria
- K1 FAIL: best gain (+8.1 sclip / +2.4 naclip) << 70% of REVA's training-free
  gain (~+22 on dev). Learned universal bg queries cannot match vocabulary-adaptive
  VABS negatives: VABS's power comes from selecting negatives RELATIVE to the target
  vocabulary (sky/grass/water are valid VOC negatives but are COCO-171 targets and
  hence excluded from any universal query set).
- K2 FAIL (in spirit): adapter on top of VABS vocab drops 52.8→46.5/44.2; official
  regime also degraded.
- K3 FAIL: held-out synonym vocab gets WORSE with the naming-invariance-trained
  adapter (26.9→23.9) and worse still with explicit name normalization (→20.7).
  Naming invariance trained on a WordNet pool does not generalize to held-out
  variants — it damages the pretrained text geometry instead.
- K4/K5 not reached (killed at K1/K3). No embedding collapse (eff-rank 128-133 vs 135).

## Scientific reading (constructive negative)
Text-side-only learned adaptation replays the audit paper's whitening negative
result at the trained level: text-space objectives (variant alignment, canonical
normalization) improve text geometry but consistently damage dense segmentation.
Combined with VABS's per-class failures, the evidence now covers three mechanisms
(linear whitening, learned adapters, learned bg queries) all failing on the text
side, while the only surviving fixes are visual/regional (SAM arbitration) or
vocabulary-adaptive (VABS). Robust naming behaviour appears not learnable in text
space alone without visual supervision.

Artifacts: /media/dell/DATA/ovss/checkpoints/lexro_v1_run{1,2,3}.pt, lexro_v2_norm.pt;
runs/lexro*.json; code lexro{,_cache,_train,_eval,_norm}.py.
