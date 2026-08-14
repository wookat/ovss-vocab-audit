# Pre-registration W11-J2: LLM canonicality judge unlocking the H2 rewriter (frozen before any judgment)

Date frozen: 2026-08-01, after W9-H2 NO-GO and before any LLM judgment is
produced.

## Question
H2's discrete word-space rewriter recovers 35-44% of non-canonical/ANS
damage but fails no-harm because it cannot tell canonical names from
non-canonical ones (geometry and frequency signals are dead). Untested
signal family: LLM world-knowledge judgment. Does an LLM canonicality
gate + rewrite satisfy no-harm while retaining the recovery?

## Frozen procedure
- Judge: a single frozen prompt, one pass, offline, recorded verbatim in
  the artifact: "For the visual concept category named '<name>', answer:
  (a) is '<name>' the everyday canonical English name a layperson would
  use for this category? YES/NO. (b) If NO, give the single most common
  everyday name. Answer as JSON {\"canonical\": bool, \"rewrite\": str|null}."
- The judge LLM is the session's own model (disclosed); judgments are
  produced once for the union of all names in the evaluated vocabularies
  and frozen before any segmentation run.
- Rewrite rule: names judged canonical are left untouched; names judged
  non-canonical are replaced by the LLM rewrite (lowercased). No CLIP
  cosine guard, no frequency tie-break (the failed H2 machinery is
  dropped entirely).
- Evaluation (identical arms to H2): SCLIP + ClearCLIP on VOC test-300
  (plain, syn100_s0, freqctrl_s0), OWLv2+SAM on VOC heldout-200 (ANS).

## Criteria (frozen, same bands as H2)
- GO: mean recovery across syn100/freqctrl/ANS arms >= 60% AND plain
  change >= -0.5 mIoU on both dense models.
- PARTIAL: recovery in [30%, 60%) with plain no-harm satisfied.
- NO-GO: recovery < 30% or plain harmed > 0.5.

## Scope guards
LLM judgments are frozen artifacts (fully released); no per-dataset
tuning; the judge never sees images or segmentation results.
