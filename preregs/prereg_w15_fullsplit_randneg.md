# W15-B: full-split 3-seed random-negative control (frozen before runs)

Trigger: REVA incremental re-review — remaining strong-accept gap "full-split
multi-seed": the 3-seed selection-advantage estimate lives on test-300, where the
seed spread is comparable to the 3.1 subset noise floor. Full dev-excluded split
(1449 minus dev-100 = 1349 images) shrinks image-sampling noise and settles whether
the VABS-vs-random selection advantage is real at scale.

## Design
probe_d1sam.py, VOC-21, offset 0 limit 1449 --skip-dev (= the paper's main-table
split), SCLIP + NACLIP, frozen VABS recipe (scene, M=64, tau 0.90, voc21_plain_vabs64).
Random arms: voc21_plain_randneg64.json (s0), _s1, _s2 (already generated; no new
tuning). 6 runs: w15_fullrand_s{0,1,2}_{sclip,naclip}.json. Comparison metric:
sam_reg_vabs minus sam_reg_rand (region-arbitration selection advantage), plus the
pixel-level counterpart as secondary.

## Frozen interpretation
- Selection advantage positive in >=5/6 cells AND mean >= +1.0 per variant:
  claim upgraded from "seeded mean with noise-floor caveat" to "consistent
  positive selection advantage on the full split" (abstract wording may drop the
  noise-floor hedge for the full-split figure, keeping the test-300 spread
  disclosure).
- Mixed signs or mean < +1.0: keep current downgraded wording; report the
  full-split numbers as-is.
- No retuning of any knob after seeing results.
