# Pre-registration W11-J1: alias-diversity hypothesis for recipe robustness (frozen before any measurement)

Date frozen: 2026-08-01, after W10/H3 (recipe ladder MIXED) and before any
corpus statistics are computed.

## Hypothesis
The synonym robustness bought by the GRIT tier in the MM-GDINO ladder is
driven by alias diversity in the training text: web grounding captions
expose a concept under many names, detection-style annotation (O365,
V3Det) under one canonical name each. Explicitly distinct from the dead
H1 frequency law: the variable is the TRAINING-side exposure diversity of
a concept's names (entropy over aliases), not the TEST-side corpus
frequency of a name.

## Measurement (frozen)
- Corpora text sources: GoldG annotation phrases (mdetr release), a fixed
  random sample of >= 200k GRIT captions (first available shard(s),
  seed 0), V3Det category name list, O365 category name list.
- Concepts: the 20 VOC foreground classes.
- Alias set per concept: the frozen WordNet cosine-filtered synonym pool
  already used by the audit (same pool as syn100/ANS candidates) plus the
  canonical name.
- Alias diversity per concept per corpus: Shannon entropy of the alias
  count distribution over that pool (0 if only one alias or absent).
- Recipe-level index: mean alias entropy over the 20 concepts of the
  tier's text mixture (T1 = GoldG; T2 = GoldG+GRIT; T3 = GoldG+GRIT+
  V3Det names).
- Class-level test: Spearman between per-class syn100 damage reduction
  from T1 to T2 (from the archived W10 confusion matrices) and the
  class's GRIT alias entropy.

## Criteria (frozen)
- GO: recipe-level alias-entropy ranking matches the robustness ranking
  (T2 >= T3 > T1 on entropy, matching drops 22.2 <= 24.7 < 32.2) AND
  class-level Spearman >= 0.4.
- NO-GO: class-level Spearman < 0.25, or GoldG vs GRIT alias entropy
  shows no substantive difference (< 10% relative), i.e. the hypothesis
  variable has no measurable contrast.
- MIXED: in between; report observationally.
- Kill for the causal upgrade (fine-tuning pair): not run in this wave.

## Scope guards
Observational; corpus sample only for GRIT; concept matching by string /
lemma match is approximate and disclosed; no training-side repair claim.
