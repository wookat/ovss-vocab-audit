# Prereg W29: vocabulary-axis falsification robustness over synonym draws (frozen before runs)

Date frozen: 2026-08-04, after W28. W27 falsified the vocabulary-level
crowding claim on ONE synonym draw (s0), and the paper says "one draw
tested". W29 tests whether the falsification pattern (synonyms lower
BOTH crowding and gain) replicates over the two other archived draws,
removing or confirming the single-draw caveat.

## Protocol (frozen; identical to W27, ViT-B/16)
- Hosts {sclip, naclip, clearclip} x {voc21, cocoobj} x draws
  {syn100_s1, syn100_s2} = 12 new cells. Plain reference cells reuse
  the W27 values (same protocol, same run family).
- S1: mean max raw cosine, 50 unlabeled images, GT never read.
- Gains: test-300, single seed, VABS-64 re-selected per vocabulary
  (new files {voc21,cocoobj}_syn100_{s1,s2}_vabs64.json).
- Note: cocoobj_syn100_s1/s2 vocab files do not exist yet (only s0);
  if absent they are generated with the SAME frozen synonym-substitution
  script and seed convention used for the existing draws before any
  evaluation is run, and archived.
## Criterion (frozen)
- 12 plain-vs-syn pairs (3 hosts x 2 datasets x 2 draws). Prediction
  under the W27 falsification pattern: syn has lower S1 AND lower gain.
- >= 9/12 pairs matching the pattern -> the falsification is
  draw-robust; the paper caveat "one draw tested" is replaced by
  "three draws tested".
- <= 6/12 -> the W27 H3 result is draw-dependent; the paper keeps the
  single-draw caveat and reports the mixed replication honestly.
- 7-8/12 -> partial; reported descriptively, caveat kept.
Either way no new claim is created: this only calibrates the caveat on
an existing negative result.
## Caveats frozen
Single seed, test-300; draws share the synonym source lexicon, so they
are not fully independent perturbations.

## AMENDMENT A (frozen 2026-08-04, before any evaluation run)
Feasibility check on the archived vocab files found that the frozen
syn100 rule is DETERMINISTIC: frac=1.0 replaces every class that has a
valid synonym with its single top-cosine WordNet synonym, so
syn100 "draws" are not distinct (byte-identity verified on
voc21_syn100_s0 == voc21_syn100_s1; cocoobj_syn100_s1/s2 were never
generated --- their determinism follows from the frac=1.0 rule, not
from a file comparison). There are no independent "draws" of the existing
rule; the original W29 design is vacuous and is amended (not run) as
follows:
- New draw rule (frozen): for each class with >= 2 valid synonyms
  (cosine in [0.70, 0.95] to the original name, same validity window
  as perturb.py), pick a UNIFORMLY RANDOM valid synonym other than the
  top-cosine one already used by syn100; classes with < 2 valid
  synonyms keep their syn100 name. Seeds 1 and 2 -> files
  {voc21,cocoobj}_synr_s{1,2}.json, generated with the sclip ViT-B
  text encoder (same as perturb.py) and archived with a manifest.
- Everything else in the protocol and the 12-pair criterion is
  unchanged (draws are now genuinely distinct vocabularies).
- The paper caveat correction is required REGARDLESS of the outcome:
  "one draw tested" is wrong wording --- the frozen syn100 rule admits
  exactly one vocabulary; the new synr draws test a strictly harder
  perturbation (second-tier synonyms, lower cosine to the original).
