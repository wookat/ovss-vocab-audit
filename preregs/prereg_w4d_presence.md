# Preregistration W4d: Presence-gated REVA (frozen 2026-08-01)

Candidate A from Ideator-5: third training-free REVA component — per-vocab-item
presence gating from SAM region evidence, with explicit abstention, targeting
the distractor-injection collapse (all-class mIoU ~4 under dis_near200).

## Method (frozen)
For each vocabulary item c: presence score s(c) = mean of top-K (K=3) SAM
region-pooled probabilities for c minus the VABS-negative margin in those
regions. Gate by within-vocabulary relative ranking: keep items with
s(c) >= tau * median(s) (tau frozen at 1.0 on dev-100 before test); pixels of
gated-out classes re-argmax over the surviving vocabulary. No absolute
thresholds, no per-dataset tuning.

## Tests (VOC-21 test-300, ClearCLIP + NACLIP bases first)
- E1 distractor axis: vocab = official + 200 near distractors. GO if all-class
  mIoU rises from ~4 to >= 20 while GT-present mIoU stays within 2.0 of the
  ungated run.
- E2 no-harm: on clean official and plain vocabs, gating must not cost more
  than 0.5 mIoU (false-rejection safety).
- E3 gate quality: report precision/recall of presence detection against GT
  presence (a class is present if >=1 GT pixel). Target recall >= 0.90.

## Kill criteria
- KILL if E1 all-class < 12 or GT-present drop > 4.
- KILL if E2 loss > 1.5 on either clean vocab (safety violation).
- If gate recall < 0.80, report as negative result (presence not decidable
  from region evidence at this granularity), do not tune past frozen tau.

## Disclosures
Uses GT presence only for evaluation, never for gating. Distinct from
ActiveSAM (2606.16996: inference speedup, clean vocab) per Ideator-5 collision
check; we target robustness + abstention semantics.
