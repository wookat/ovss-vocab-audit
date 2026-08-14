# Pre-registration W9-H2: name canonicalization repair go/no-go (frozen before run)

Date frozen: 2026-08-01, after the W9-H1 NO-GO (disclosed: the frequency
mechanism died, so the hypothesis here is canonical-name restoration, not
frequency-aware renaming).

## Question
Can a training-free, model-agnostic DISCRETE-word-space repair — rewriting
each user-supplied name to a canonical alias before inference — recover the
synonym-axis damage, without hurting plain vocabularies? This is the only
untested repair family (embedding-space, prediction-space, and
training-side repairs are all preregistered kills).

## Method (frozen)
For each incoming name: candidate alias set = the name itself + WordNet
synonyms of its first sense lemma + head noun of multiword names.
Canonical pick = the candidate with the highest wordfreq zipf score,
tie-break by shorter CLIP BPE token count, subject to a semantic-equivalence
guard: CLIP text cosine between "a photo of a {name}." and
"a photo of a {candidate}." >= 0.80 (below that, keep the original name).
No LLM (offline constraint disclosed). No per-dataset tuning; thresholds
frozen here.

## Evaluation (frozen)
- Models: SCLIP, ClearCLIP (dense) on VOC test-300; OWLv2+SAM box harness
  on VOC heldout-200 (for the ANS defense arm).
- Arms: (1) syn100_s0 vocabulary, canonicalized vs raw, vs plain;
  (2) freqctrl_s0 (rare controls), canonicalized vs raw;
  (3) ANS vocabulary on OWLv2, canonicalized vs raw (defense test);
  (4) plain vocabulary, canonicalized vs raw (no-harm check).
- Recovery = (canon - raw) / (plain - raw) on each damaged arm.

## Criteria (frozen)
- GO: mean recovery across arms (1)-(3) >= 60% AND plain arm change
  >= -0.5 mIoU on both dense models -> promote to full matrix + REVA third
  component + paper section.
- PARTIAL: recovery in [30%, 60%) -> report as a partial repair row in the
  benchmark, no standalone claim.
- NO-GO: recovery < 30% or plain harmed > 0.5 -> repair family recorded as
  killed; the discrete word space joins the graveyard.

## Cost
~8 runs, < 1 GPU day.
