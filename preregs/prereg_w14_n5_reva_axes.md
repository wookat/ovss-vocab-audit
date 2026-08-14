# W14-N5: REVA under perturbed vocabularies + seeded random-negative control (frozen 2026-08-02, before any run)

Trigger: blind-review round W14 — REVA R1 (Weak Reject) demands (Q1) REVA's
behaviour on the synonym and searched (ANS) axes before the title claim
"vocabulary-robust" can stand, and (Q2) >=1 extra seeds for the
random-negative control.

## Part A: REVA on syn100 and ANS vocabularies
Design: SCLIP + NACLIP, VOC-21 test-300, probe_d1sam arms
(pix_vabs / sam_reg_vabs / sam_reg_rand), where the base vocabulary is
(a) voc21_syn100_s0 and (b) the archived ANS searched vocabulary; VABS
negatives regenerated conditioned on each perturbed vocabulary
(vabs.py, scene lexicon, M=64, tau=0.90 — frozen recipe, no re-tuning).
Reference points: the corresponding pixel baselines of each vocabulary
(existing archived runs where available, else the pix arm of this run).

Frozen expectations (descriptive, no GO/NO-GO — this is a scope-setting
measurement): report REVA delta over each perturbed-vocabulary pixel
baseline. Honest write-up rules fixed now:
- If REVA adds a comparable macro gain on syn100 as on plain, the axis
  can be claimed as covered.
- If the gain shrinks or reverses (expected: the audit shows the synonym
  drop is inter-class confusion, which background negatives cannot fix),
  the title/abstract scope is narrowed to plain-vocabulary robustness
  and the numbers are reported as a disclosed boundary.
- ANS is reported as a stress bound in either case.

## Part B: random-negative control extra seeds
randneg64 seeds 1 and 2 generated with the same recipe; SCLIP + NACLIP,
VOC test-300, sam_reg_rand arm. Report mean +/- spread of the selection
advantage (sam_reg_vabs - sam_reg_rand) over the 3 seeds. No verdict
change unless the seed spread swallows the advantage (then §4.2's
selection-advantage claim gets a noise-floor caveat).

Notes: test-300 (full-split multi-seed deferred to camera-ready
compute); single dataset; frozen VABS hyper-parameters.
