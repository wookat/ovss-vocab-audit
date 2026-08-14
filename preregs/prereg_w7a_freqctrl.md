# Pre-registration W7a: frequency/token-matched control for ANS cross-paradigm transfer (frozen before run)

Date frozen: 2026-08-01. Source: R2 incremental review item M2 on the W6
cross-family section.

## Question
Is the ANS vocabulary's transfer damage on OWLv2 (-31.8 held-out) explained
by adversarial structure found by the search, or merely by ANS names being
low-frequency / long-tokenization synonyms (a property any rare synonym
shares)?

## Control construction (frozen)
For each VOC class where ANS changed the name, sample a DIFFERENT WordNet
synonym of the same class, drawn from the same candidate pool used by ANS
(CLIP cosine to plain name in [0.70, 0.95]), matched to the ANS choice on
CLIP BPE token count (+/-1 token; if no match exists, nearest token count).
Classes ANS left unchanged stay unchanged. 3 seeds. If a class has no
alternative candidate, keep the ANS name for that class (disclosed count).

## Experiment (frozen)
Evaluate the 3 control vocabularies on: (a) OWLv2+SAM harness, VOC-21
held-out-200 (offset 100), same THRESH=0.2; (b) ClearCLIP pixel baseline,
same split. Compare: plain, ANS, mean of 3 controls.

## Interpretation (frozen)
- ADVERSARIAL: mean control drop <= 50% of the ANS drop on OWLv2
  (i.e. controls lose <= 15.9 where ANS loses 31.8) -> ANS transfer is
  search structure, headline claim stands.
- RARE-SYNONYM: mean control drop >= 80% of ANS drop -> demote the claim to
  "rare-synonym sensitivity"; the paper wording changes accordingly.
- MIXED: in between -> report both components with the split.

## Cost
3 control vocabs x 2 harnesses x 200 images: ~2-3 GPU hours.
