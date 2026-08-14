# W16-B: SC-CLIP author-code anchor (frozen before runs)

Trigger: audit final review camera-ready item — SC-CLIP is the only leaderboard
row without an author-code anchor (ProxyCLIP: NEG +14.0 vs local +22.1;
LPOSS: NEG +6.0 vs local +19.6 already anchored).

## Design
Unmodified official SC-CLIP release (TIP'25, github.com/SuleBai/SC-CLIP),
its own protocol (mmseg 1.1.1 pipeline, its VOC config), full VOC val
('voc21' background task). Only the class-name list is changed:
1. official: repo's shipped class names (reproduce published ~59-60 VOC number);
2. plain: our voc21_plain.json names;
3. syn100: our voc21_syn100_s0.json names.
Env: new conda env per repo README (torch 1.10.1+cu111, mmcv 2.0.1,
mmseg 1.1.1) on temp-hb. HF not needed (openai CLIP weights; download locally
and transfer if remote fetch fails).

## Frozen interpretation
- Direction replicates if official - plain (NEG) > 0 and syn100 < plain.
- NEG magnitude is expected to be protocol-conditional (as with ProxyCLIP/LPOSS);
  whatever the value, the paper's SC-CLIP rows are re-anchored to the
  author-code NEG for leaderboard-size claims and the "SC-CLIP unanchored"
  caveat is removed.
- If the official number fails to reproduce within 1.5 mIoU of the published
  VOC value, report the anchor as failed-reproduction and keep the caveat.
- No tuning of any repo knob; only the name list changes.
