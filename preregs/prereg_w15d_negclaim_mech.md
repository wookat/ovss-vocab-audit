# W15-D: negative-claim mechanism accounting (frozen before runs)

Trigger: REVA incremental re-review — strong-accept gap "mechanism explanation":
why does vocabulary-conditioned VABS selection beat matched random negatives under
region arbitration? Hypothesis (stated before measurement): VABS negatives claim
more of the TRUE background (higher recall on GT-background pixels) without
claiming more foreground (equal or better precision); random negatives either
under-claim background (leaving it to leak into objects) or over-claim foreground.

## Design
Observational accounting on VOC-21 test-300, SCLIP + NACLIP, frozen recipe,
same arms as probe_d1sam (vabs vs rand-s0 negatives, region arbitration).
For each arm log, over all pixels: (a) share of pixels assigned to any negative
(= predicted background), (b) precision of negative claims against GT background,
(c) recall of GT background, (d) foreground pixels wrongly claimed by negatives
(absolute count). Runs: w15d_mech_{sclip,naclip}.json.

## Frozen interpretation
- Hypothesis SUPPORTED if VABS's GT-background recall exceeds rand's by >= 3 points
  while its precision is within 2 points of (or above) rand's, in both variants.
- If VABS instead wins by claiming LESS foreground (precision up, recall equal),
  report that as the mechanism (still a positive finding, different wording).
- If neither pattern holds (mixed across variants), report as descriptive
  accounting without a mechanism claim.
- Descriptive, single seed, test-300; no knob changes.
