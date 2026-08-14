# Pre-registration W12-K4: adaptive-attack check of detector-guided pruning (frozen before any run)

Date frozen: 2026-08-01, before any pruning-under-ANS evaluation.

## Question
If J5 pruning becomes a REVA component, does it survive the strongest
existing naming attack? Re-evaluate the archived ANS-searched vocabulary
(searched on ClearCLIP, held-out split) under the frozen J5 pruning rule
(OWLv2 base-ensemble, "a photo of a X", threshold 0.2, background kept).

## Frozen design
VOC heldout-200 (same split as the original ANS evaluation), models
SCLIP and ClearCLIP, vocabularies: ANS vocabulary; matched rare-synonym
control (seed 0) for reference. Arms: dense vs dense+pruning.

## Criteria (frozen)
- DEFENDED: pruning recovers >= 60% of the ANS damage (relative to each
  model's plain baseline) on both models -> report as positive
  robustness property; a detector-aware search becomes worthwhile
  follow-up before any strong claim.
- NOT DEFENDED: recovery < 30% on either model -> disclose that pruning
  does not mitigate worst-case naming attacks (expected if the attack
  damages the CLIP scoring of present classes rather than adding absent
  winners).
- MIXED: between.

Either outcome enters the REVA J5 subsection as an honest
adaptive-attack disclosure; no threshold tuning.

## Addendum (frozen in same commit): ADE-150 scale cell
J5 pruning on ADE-150 plain vocabulary, SCLIP + NACLIP, test-300.
GO: mean gain >= +2 with no model harmed >1; NO-GO: mean < +1 or any
harm > 3; MIXED between. Purpose: does the gain grow with vocabulary
size (absent-class count) as the mechanism accounting predicts at the
corpus level?
