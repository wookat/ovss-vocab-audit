# W15: LPOSS author-code anchor (frozen before runs)

Trigger: blind-review convergent must-fix — leaderboard/ranking statements about the
LPOSS-style row rest on an unanchored style-reimplementation. Mirror of the ProxyCLIP
author-code anchor.

## Design
Unmodified official LPOSS release (CVPR'25, github.com/vladan-stojnic/LPOSS),
lposs.yaml config, its own protocol (mmseg pipeline, 2048x448 slide 224/112, DINO+
MaskCLIP backbones), full VOC val ('voc' 21-class task). Only the class-name list is
changed via an external wrapper (w15_lposs_anchor.py; author code untouched):
- Arm 1 official: authors' shipped CLASSES (includes the 26-term background expansion).
- Arm 2 plain: our voc21_plain.json (background + 20 plain names).
- Arm 3 syn100: our voc21_syn100_s0.json (same substitution seed as the audit suite).

## Frozen interpretation
- Direction replicates if official - plain (NEG) >= +8 mIoU and official - syn100 shows
  a synonym drop (plain - syn100 or official-relative) of >= 2.
- Effect sizes are expected to differ from our-protocol numbers (as with ProxyCLIP,
  where author-code NEG was +14.0 vs our +22.1); differences are reported, not tuned.
- If NEG < +8: the LPOSS-style leaderboard statements are downgraded/flagged in the
  audit paper rather than defended.
- Any failure to run (env/weights) is reported as infrastructure, not evidence.
