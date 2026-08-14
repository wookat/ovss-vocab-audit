# Pre-registration W13-L5: is VABS beaten by a scalar background boost? (self-attack, frozen before any run)

Date frozen: 2026-08-01, after W13-L3's boost control (which used
separate-entry fillers and eval-set-tuned boosts) and before any
fold-convention, dev-tuned comparison.

## Question
W13-L3 showed a background-logit boost of +0.02 cosine units reaches
47.5 GT-present mIoU on SCLIP/VOC — above the archived pix-VABS 41.2.
If a one-parameter boost matches or beats VABS negatives under matched
conventions and honest tuning, the REVA paper's VABS component must be
reframed: its generic-absorption share is reproducible by calibration,
and only the selection advantage over random negatives (+1.3 to +4.2)
is attributable to design.

## Frozen design
Models: SCLIP + NACLIP. Dataset: VOC, images 0-99 = dev (boost
selection), images 100-299 = eval. Convention: identical GT-present
mIoU, 21-class output space (boost applied to the background logit;
VABS negatives folded into background as in the REVA pipeline).
Arms on eval split:
1. plain;
2. plain + background boost b* (b* = argmax over
   {0.005,0.01,0.02,0.03,0.05,0.08} on dev split only);
3. pix-VABS-64;
4. pix-VABS-64 + boost b** (re-selected on dev with VABS active) —
   are the effects additive or the same mechanism?
Also the SAM arm for SCLIP only: sam_reg_vabs vs sam_reg with boost.

## Criteria (frozen)
- VABS-SUBSUMED: arm2 >= arm3 - 0.5 AND arm4 <= arm3 + 1.0 (boost
  replicates VABS and VABS adds nothing on top of boost) -> REVA paper
  must reframe VABS as background recalibration with a small selection
  term; disclose prominently.
- VABS-DISTINCT: arm2 <= arm3 - 2.0 OR arm4 >= max(arm2, arm3) + 2.0
  (VABS clearly beats honest boost, or adds on top of it) -> add the
  boost as a disclosed baseline; VABS claim stands.
- MIXED otherwise: report both numbers, add boost baseline row, soften
  the VABS mechanism language.

## Scope
VOC only at go/no-go; any reframing requires ADE/Context replication
(where VABS was already null — consistent with the calibration story).
Verdict applies to the text-side VABS component, not SAM arbitration.

## Addendum (frozen 2026-08-01 before any cross-dataset run; triggered by R1 must-fix)
Cross-dataset transfer of the VOC-dev-selected boost scalar (SCLIP
b*=0.03, NACLIP b*=0.05), applied unchanged to ADE-150 and Context-60
plain (test-300), vs plain and vs the pixel-VABS arm on the same split.
- TRANSFERS: boost gains >= +1 GT-present mIoU on both datasets for
  both models -> the boost is a portable competitor; "label-free"
  advantage of VABS must be weakened to "no per-vocabulary re-tuning".
- NOT-TRANSFER: boost changes < +0.5 or harms on any dataset/model ->
  the scalar is dataset-specific; VABS's advantage includes
  transfer-without-retuning and the disclosure says so.
- MIXED otherwise.
Note: VABS was already measured null on ADE (W2a) — the comparison is
still informative: a boost that HARMS on ADE is worse than a null VABS.
