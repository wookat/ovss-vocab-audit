# Preregistration W2e: Robust-mIoU benchmark go/no-go (frozen 2026-07-31)

## Question
Do newer (2024-25) training-free OVSS methods still carry a large naming
engineering gain (NEG = official-vocabulary mIoU minus plain-vocabulary mIoU)
and synonym fragility, i.e. would a vocabulary-robustness benchmark reshuffle
their ranking? Go/no-go on the D (Robust-mIoU) direction.

## Setup (frozen)
- First new baseline: ProxyCLIP-style proxy attention (arXiv:2408.04883)
  reimplemented in our unified pipeline: DINO patch-affinity proxy attention
  over CLIP value features at the last block. DINO ViT-S/16 (not the paper's
  ViT-B/8 — weights availability), CLIP ViT-B/16 openai, unified protocol
  (336/224/112, scale 40). Disclosed as a style-reimplementation, not a
  reproduction; its official-vocab mIoU is reported next to the published
  number for context only.
- VOC-21 test-300 (images 0-299), vocabularies: plain, official, syn100_s0.
- NEG = official - plain; synonym drop = plain - syn100.

## Go / kill criteria (frozen)
- GO if ProxyCLIP NEG >= 3 mIoU OR synonym drop >= 3 mIoU (vocabulary
  sensitivity persists in the DINO-guided generation) -> proceed to add 2 more
  2024-25 baselines (CLIPer, TraceCLIP or LPOSS) and full benchmark prereg.
- KILL the reshuffle story if NEG < 3 AND synonym drop < 3: DINO guidance
  already washes out naming sensitivity -> pivot to short negative-result note.
- Sanity bar: our ProxyCLIP-style official mIoU must be within 15 mIoU of the
  published VOC number (~80 with bg-heavy protocol differences aside); if
  wildly off, fix implementation before reading criteria (implementation
  validity gate, not a result criterion).
