# W14-N3: fixed hand list vs VABS on non-VOC vocabularies (frozen 2026-08-02, before any run)

Trigger: rebuttal drill Attack 1 — on VOC the SCLIP 26-entry hand-written
background list matches or beats VABS under identical SAM pooling; no
experiment shows the adaptive component succeeding where the static list
fails. VABS's claimed differentiator (vocabulary adaptivity) is untested
outside VOC.

Design: SCLIP + NACLIP, pixel-level arms (no SAM; isolates the
text-side/background mechanism), test-300, frozen protocol unchanged.
Datasets/vocabs: ade150_plain (150 cls, no bg row) and cocoobj_plain
(80 things + background). Arms per cell:
1. plain (reference, existing runs OK)
2. plain + SCLIP 26-entry hand list appended as extra background rows
   (folded to background channel on COCO-Object; on ADE-150 folded to an
   added "ignore/none-of-the-above" background channel exactly as VABS
   extra rows are handled — same folding convention as the vabs64 arm)
3. plain + VABS-64 (existing vabs64 vocabularies)

Frozen verdicts (per-dataset, GT-present mIoU, mean of 2 models):
- HANDLIST-DOMINATES: hand list >= VABS - 0.3 on BOTH datasets -> the
  adaptive component has no demonstrated value beyond VOC either; REVA
  must be reframed (VABS demoted to convenience; arbitration + any
  background sink is the story). Write it in.
- VABS-ADAPTIVE: VABS > hand list + 0.5 on at least one dataset ->
  first positive evidence for adaptivity where the VOC-shaped list's
  assumptions break; add to REVA §4.2 and pre-empt Attack 1.
- MIXED otherwise: report both numbers with no adjudicated winner.

Notes: single seed, test-300; ADE VABS is a disclosed null (W2a) — if
both hand list and VABS are null on ADE, the COCO cell decides. The
hand list is used verbatim (SCLIP official VOC background expansion,
26 entries); no similarity filtering is applied to it (that is the
point: it is the zero-effort baseline).
