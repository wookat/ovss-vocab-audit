# Pre-registration W13-L3: vocabulary-filler free lunch — pressure-valve mechanism (frozen before any run)

Date frozen: 2026-08-01, before any pseudo-word run.

## Hypothesis (eleventh-round ideator L3)
The K2 observation — injecting random absent names raises GT-present
mIoU by +6..+9 from the first name — is a semantic-free softmax
pressure-valve effect: background pixels that dense CLIP must assign to
SOME foreground class get absorbed by innocuous filler entries. If the
mechanism is pressure release rather than semantics, meaningless
pseudo-words should reproduce most of the gain, and the pixels the
fillers capture should be overwhelmingly GT-background.

## Frozen design
VOC test-300, SCLIP (go/no-go model; replicate on NACLIP if GO).
Arms, each n=50 filler entries appended to plain VOC-21:
- R: random real nouns (the K2 rand pool, first 50);
- P: pseudo-words — pronounceable nonsense strings (CVCV patterns,
  frozen seed 0, no dictionary hits);
- V: the VABS-64 negatives (first 50) as the designed-negative
  reference.
Metrics: GT-present mIoU delta vs plain; filler-captured pixel
decomposition by GT row (background vs foreground share).
Trivial-baseline control: background-logit additive boost swept over
{+0.01, +0.02, +0.05} cosine units on plain vocabulary (no fillers) —
does directly boosting the background class reproduce >= 85% of the
R-arm gain?

## Criteria (frozen)
- GO (pressure-valve mechanism): pseudo-word arm P achieves >= 70% of
  the R-arm gain AND filler-captured pixels are >= 80% GT-background
  AND the background-boost trivial control reproduces < 85% of the
  R-arm gain.
- NO-GO: P < 40% of R (effect requires semantics), OR trivial
  background boost >= 85% of R (it is just background calibration in
  disguise).
- MIXED otherwise.

## Scope
One model, one dataset at go/no-go. GO requires NACLIP + COCO
replication before any claim. Regardless of verdict, results are
recorded; a GO feeds a VABS-mechanism paragraph in the REVA paper
(how much of VABS is generic absorption vs designed negatives), not a
standalone paper unless the cross-dataset law holds.
