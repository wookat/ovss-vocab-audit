# Prereg W39: full-split replication of the remaining subset-based key cells (frozen before runs)

Date frozen: 2026-08-25, before any W39 evaluation is launched.

Motivation (reviewer risk): two REVA boundary/transfer claims still rest
on 300-image subsets whose image-resampling noise floor is 3.1 mIoU:
(a) the ADE-150 no-background boundary cells (W2a, val first-300), and
(b) the ViT-L/14 second-backbone grid (W4i, VOC test-300). W39 re-runs
these cells at full-split scale with every vocabulary file, checkpoint,
and protocol constant reused verbatim from the archived runs. No new
mechanism, no tuning; this batch can only confirm, soften, or annotate
existing paper sentences.

## Protocol (frozen)
Shared: unified pipeline (clip_seg/eval_seg/data), ViT-B/16 defaults
(short 336, slide 224/112, logit scale 40) unless stated; SAM ViT-B
sam_vit_b_01ec64.pth, points_per_side 16; vocab files reused verbatim
from perturbed_vocabs/ (no regeneration); single pass (all arms are
deterministic given the archived vocab files; the rand.64 arm uses the
same archived seed-0 set as the original runs).

Part A — ADE-150 full val (2000 images, offset 0), hosts {sclip, naclip}:
- plain pixel reference: run_eval.py --vocab-file ade150_plain.json
- pix_vabs / sam_reg_vabs / sam_reg_rand: probe_d1sam.py
  --vabs-vocab ade150_plain_vabs64.json --rand-vocab ade150_plain_randneg64.json
Published first-300 values being replicated: pooling gain +1.04/+1.32
over pix_vabs; VABS null vs plain (+0.05/−0.65) and ≈ random.

Part B — VOC-21 dev-excluded full split (1349 images, --skip-dev),
model ViT-L-14-quickgelu (openai), hosts {sclip, naclip}:
- plain pixel: run_eval.py --vocab-file voc21_plain.json
- official pixel: run_eval.py (dataset default names)
- pix_vabs / REVA (sam_reg_vabs) / rand.64+SAM (sam_reg_rand):
  probe_d1sam.py --vabs-vocab voc21_plain_vabs64.json
  --rand-vocab voc21_plain_randneg64.json
Published test-300 values being replicated: SCLIP plain 37.24 → REVA
44.01 (official pixel 40.72); NACLIP 36.31 → 51.60 (50.60).

## Criteria (frozen; every outcome is written into RESULTS and the paper)
- H-A1 (pooling transfers to the no-background regime):
  sam_reg_vabs − pix_vabs ≥ +0.5 on BOTH hosts → the paper's
  "+1.0/+1.3, val first-300" sentence is upgraded to the full-val
  numbers. Any host < +0.5 → the first-300 sentence is kept and the
  full-val value is added with an explicit "does not replicate at
  full scale" disclosure.
- H-A2 (VABS null boundary): |pix_vabs − plain| ≤ 2.0 AND
  |sam_reg_vabs − sam_reg_rand| ≤ 2.0 on both hosts → boundary claim
  ("VABS has no room without a background class") confirmed at full
  val and the paper cites the full-val cells. A violation > +2.0
  (VABS beats plain) weakens the boundary claim and must be written
  into sec:transfer; a violation < −2.0 is disclosed as harm.
- H-B1 (ViT-L transfer direction): REVA − plain ≥ +3.0 on both hosts →
  "the pipeline transfers to ViT-L" sentence keeps its strength with
  full-split numbers. Any host < +3.0 → sentence weakened to the
  measured value.
- H-B2 (above-official-pixel claim): REVA ≥ official_pixel − 0.5 on
  both hosts → the "exceeds the pixel-level official reference"
  sentence is upgraded to full split. Any host below → the sentence is
  restricted to the host(s) where it holds, or dropped if none.
- Descriptive only (no claim, single seed): VABS-vs-rand under
  arbitration on both parts.

## Caveats frozen
Single seed on the random-negative arm (archived seed-0 set; the
3-seed replication of finding (ii) exists only for ViT-B and is not
extended here); ViT-L VABS negatives remain the archived ViT-B-selected
set (transfer, already disclosed in the paper); SAM ViT-B pps16
throughout; no per-class safety analysis in this batch.
