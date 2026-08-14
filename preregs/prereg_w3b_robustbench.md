# Preregistration W3b: Robust-mIoU benchmark round 1 (frozen 2026-07-31)

## Goal
Quantify vocabulary robustness across training-free OVSS generations and test
whether a robustness-aware ranking reshuffles the official-vocabulary ranking.

## Methods (7, unified protocol ViT-B/16 openai, 336/224/112, scale 40)
- MaskCLIP, SCLIP, ClearCLIP, NACLIP (existing).
- ProxyCLIP-style (W2e, DINO ViT-S/16 proxy attention).
- LPOSS-style (arXiv:2503.19777): DINO-affinity label propagation over the
  patch-level class logits of the strongest base (kk/NACLIP-style exit);
  alpha=0.9, 10 iterations, k=32 neighbours.
- SC-CLIP-style (arXiv:2411.15869): anomaly-token restoration (patch tokens
  whose pre-block L2 norm > 5x median are replaced by 3x3 neighbour mean)
  followed by kk attention, no residual/FFN.
All three new rows are style-reimplementations in our pipeline, not
reproductions; published numbers cited for context only.

## Vocab suite (frozen; existing generators/seeds)
official, plain, syn50 s0/s1/s2, syn100 s0, dis_near200 (GT-present scoring
for the distractor cell as in the audit paper).

## Metrics (frozen definitions)
- robust-mIoU = mean over {plain, syn50 s0-2, syn100 s0}.
- worst-case robust-mIoU = min over the same set.
- NEG = official - plain.
- Reshuffle statistic: Kendall tau between the 7-method ranking by official
  mIoU and by worst-case robust-mIoU.

## Data
VOC-21 test-300 (round 1); extension to Context-60/COCO-Object in round 2 if
round 1 supports the story.

## Go / kill (frozen)
- GO for full benchmark paper if (a) at least 2 of the 3 new-generation rows
  have NEG >= 3 or synonym drop >= 3, AND (b) Kendall tau < 1.0 (at least one
  rank swap between official and worst-case rankings).
- If all new rows have NEG < 3: pivot to negative-result note ("the newest
  generation fixed naming engineering").
- Sanity gates per new row: official mIoU within 15 of published protocol-
  comparable numbers; else fix implementation before reading criteria.
