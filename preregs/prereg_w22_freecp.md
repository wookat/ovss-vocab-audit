# Prereg W22: FreeCP (ICCV 2025) author-code anchor + distractor interaction (frozen before runs)

Date frozen: 2026-08-03 (before any W22 arm is launched)

## Motivation
FreeCP (ICCV 2025, arXiv 2508.00557, official repo chenqi1126/FreeCP) is a
training-free *class purification* plugin: it manipulates the class/query
set itself before dense classification. It is therefore the first audit
subject whose mechanism operates on the same object our audit perturbs
(the vocabulary). Two frozen questions:
1. Audit axes: do NEG and synonym fragility replicate on the official
   FreeCP+SCLIP stack? (seventh author-code anchor candidate)
2. Interaction: does class purification absorb distractor injection ---
   i.e., does FreeCP reduce the mIoU drop caused by +200 injected
   distractor words relative to the same base without FreeCP?

## Protocol (frozen)
- Official FreeCP repo, forward unmodified; only class-name files changed
  (and the FreeCP on/off toggle as shipped by the repo, if present; if the
  repo has no vanilla-SCLIP config, the no-FreeCP contrast is taken from
  our SCLIP in-protocol runs and labelled protocol-mixed, descriptive
  only).
- Base method: SCLIP (best-documented FreeCP pairing; published
  SCLIP+FreeCP VOC-21 65.8).
- Dataset: VOC-21 (primary). COCO-Object optional if time permits
  (published 37.2).
- Arms (VOC-21, FreeCP+SCLIP): official, plain, syn100,
  plain+dist200-near (frozen W-series near-stratum 200 distractor list,
  deterministic first-200, same file as the audit's s0 sampling).
- Contrast arm (if repo supports FreeCP off): SCLIP-only plain and
  SCLIP-only plain+dist200-near, same repo, same data.
- Reproduction gate: official arm within 1.5 mIoU of 65.8 (VOC-21). If it
  fails, no interpretation of perturbation arms on that dataset.
- Single run per arm.

## Frozen expectations / criteria
- E1 NEG > 0 and syn100 < plain (direction only).
- E2 (interaction): let D_f = plain - (plain+dist200) with FreeCP, and
  D_b = same without FreeCP (repo contrast if available, else in-protocol
  SCLIP descriptive). Verdict ABSORBS if D_f <= 0.5 * D_b; PARTIAL if
  D_f < D_b but > 0.5 * D_b; NONE otherwise.
- Interpretation guard: if FreeCP's purification simply deletes injected
  distractor names, this is a *definitional* absorption --- report which
  distractors survive purification (count) alongside the mIoU verdict, so
  absorption is attributed to name deletion vs re-weighting honestly.

## Outcome handling
All outcomes reported; failures preserved. If FreeCP absorbs distractors,
this is a positive finding FOR the audit (a vocabulary-side defense exists
and can be measured by our protocol); if not, the collapse phenomenon is
robust even under purification. Either way the result goes to the audit
paper's distractor section as a bounded note, not a new headline claim.

## FEASIBILITY ABORT (2026-08-03, before any run was launched)
Inspection of the official repo shows FreeCP's forward requires
per-class-name LLM descriptions from precomputed dicts
(prompts/vicuna13b/*.py, keyed by the official engineered names only;
get_prompts(use_llm=True) does dataset_prompts[name]). Any perturbed
vocabulary (plain/syn100/distractor) raises KeyError: the released method
cannot express a non-official vocabulary without regenerating LLM assets,
which would violate the frozen only-change-class-name-files protocol.
W22 is therefore aborted at the feasibility gate with zero runs; no
criteria were evaluated. Finding retained for the audit paper as a bounded
note: this ICCV'25 vocabulary-purification method is *closed* over its
engineered names + precomputed LLM descriptions -- vocabulary engineering
one level deeper than class-name files.
