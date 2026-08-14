# W16-C: VABS on unmodified author code (frozen before runs)

Trigger: REVA rebuttal drill open item — "does the gain survive an official
code stack?" Full REVA (SAM arbitration) cannot be plugged into author code
without modification, but the VABS component can: SC-CLIP's name file supports
comma-separated aliases, so appending the 64 frozen VABS negatives to the
background row is a pure name-file change (same rule as our pixel-level VABS
arm). SC-CLIP is the anchor with the fastest verified official run (W16-B).

## Design
Unmodified official SC-CLIP, full VOC val 1449, its own protocol. Three name
files, all with plain foreground names:
1. plain (already run, W16-B: 42.97);
2. plain + VABS-64 negatives appended to background row
   (from voc21_plain_vabs64.json row 0, unchanged);
3. plain + 64 matched random negatives (voc21_plain_rand64_s0.json row 0 or
   equivalent frozen random set used in W15-B seed 0), same budget.
No knob of SC-CLIP is changed.

## Frozen interpretation
- VABS transfer holds if (2) - (1) >= +3 mIoU on author code.
- Selection advantage holds if (2) - (3) > 0.
- If (2) - (1) < +3: report as author-code attenuation of the pixel-level
  VABS gain; REVA's our-protocol numbers stay qualified as protocol-conditional.
- Scope: this anchors the VABS component only, not SAM arbitration; stated as
  such in the paper.
