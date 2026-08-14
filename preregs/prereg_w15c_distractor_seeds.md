# W15-C: distractor-set multi-seed control (frozen before runs)

Trigger: audit incremental re-review — strong-accept gap "distractor multi-seed":
the dis_near200 condition uses the deterministic first-200 of the sorted 638-name
near stratum (a single arbitrary draw). Question: is the distractor collapse (and
its GT-present convention gap) stable across which 200 near distractors are drawn?

## Design
Generate voc21_dis_near200_s{1,2}.json: random.Random(seed).sample of 200 from the
frozen near stratum (cos in (0.80, 0.92]), same base plain vocabulary, same filter;
existing dis_near200 is the s0/deterministic draw. Evaluate SCLIP + NACLIP,
VOC-21 test-300 (offset 0 limit 300 of the eval order used by the audit's
test-300 distractor rows), reporting both all-class (fixed-denominator) and
GT-present mIoU. 4 new runs (2 methods x 2 seeds); compare to the archived
s0 values.

## Frozen interpretation
- Stable if all-class collapse values across the three draws lie within 2 mIoU
  of each other per method AND the GT-present values within 3: the paper adds
  "stable across distractor draws (3 draws)" to the distractor section.
- Larger spread: report the spread as-is and qualify the collapse magnitude as
  draw-dependent.
- No changes to the filter, strata boundaries, or metric conventions.
