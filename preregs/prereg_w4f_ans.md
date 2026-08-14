# Preregistration W4f: ANS — adversarial naming search (frozen 2026-08-01)

Ideator-5 candidate C. Claim to test: the random-synonym suite understates
vocabulary fragility; a small greedy search over per-class naming choices finds
substantially worse legal vocabularies, giving the benchmark a true worst-case
axis (analogous to adversarial attacks vs random noise).

## Method (frozen)
- Search space: for each of the 20 VOC object classes, 6 candidate names
  (plain name + 5 WordNet/synonym-bank alternatives already in our generator
  pool). Background phrase fixed to "background".
- Search: greedy coordinate descent, 1 pass over classes in fixed order
  (alphabetical), each step picks the candidate minimizing mIoU on a 100-image
  SEARCH subset (images 0-99 of the eval split). Budget: 20x6 = 120 evals of
  100 images per method.
- Evaluation: the found vocabulary is then evaluated on the DISJOINT test-200
  (images 100-299). Overfitting control: report search-subset vs held-out gap.
- Methods: ClearCLIP and LPOSS-style (weakest and most robust of the family).

## GO / KILL (on held-out test-200)
- GO if ANS vocabulary is >= 3.0 mIoU below the worst random suite member
  (syn100_s0 equivalent on the same 200 images) for both methods.
- KILL if < 1.5 on either method (search adds nothing over random suite).
- Partial (one method passes): report as method-dependent, no headline claim.

## Disclosures
All names are legal synonyms a user could plausibly type (no adversarial
gibberish); the search uses GT labels and is an upper-bound stress instrument
for the benchmark, not a deployable attack. Search/eval subsets disjoint.
