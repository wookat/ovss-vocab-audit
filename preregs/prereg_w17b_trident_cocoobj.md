# W17-B: official Trident anchor on COCO-Object (frozen before runs)

Second dataset for the 4th (SAM-coupled) author-code anchor, and a second
author-code test of the gap-vs-recovery boundary on an independent method.
Unmodified official Trident, cfg_coco_object.py (--sam_refine), full
COCO-Object val 5000 (the same converted dataset used for W16-D, built with
SC-CLIP's own mapping; both repos use the identical layout and label space).
Only the class-name file changes (repo's '; ' convention):
1. official: repo's cls_coco_object.txt (must reproduce published ViT-B/16
   COCO-Object 41.1 within 1.5 mIoU, else failed-reproduction, stop);
2. plain: cocoobj plain names;
3. syn100: cocoobj_syn100_s0 names;
4. plain+VABS64: the W16-D VABS negatives appended to background;
5. plain+rand64: the W16-D matched random negatives.

## Frozen interpretation
- Audit axes replicate if NEG = (1)-(2) > 0 and (3) < (2).
- VABS transfer holds if (4)-(2) >= +3; selection advantage if (4)-(5) > 0.
- Boundary prediction (stated before running): if Trident's COCO-Object NEG
  is small (VOC-like +3), the W16-D pattern predicts VABS ~ random (selection
  NULL); if NEG is large, the VOC pattern predicts a positive selection
  advantage. Either outcome is informative; neither will be reclassified.
- No repo knob changed; single run, single negative seed (disclose).
