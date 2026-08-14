# W16-D: SC-CLIP author-code anchor on a second dataset (COCO-Object) (frozen before runs)

Trigger: the W16-B/C author-code anchors cover VOC only; the audit's
cross-dataset synonym claims and REVA's COCO rows are local-protocol only.

## Design
Unmodified official SC-CLIP, cfg_coco_object.py, full COCO-Object val (5000
images; masks converted from cocostuff val annotations with the repo's own
clsID_to_trID mapping, val split only). Only the class-name file changes:
1. official: repo's cls_coco_object.txt (reproduce published 37.7 within 1.5);
2. plain: cocoobj_plain.json names (row 0 = background);
3. syn100: cocoobj_syn100_s0.json;
4. plain+VABS: cocoobj_plain_vabs.json negatives appended to background row;
5. plain+rand: cocoobj_plain_randneg64.json negatives, same budget.

## Frozen interpretation
- Audit axes replicate if NEG = official - plain > 0 and syn100 < plain.
- VABS transfer holds if (4)-(2) >= +3; selection advantage holds if (4)-(5) > 0.
- Failed official reproduction (>1.5 off 37.7): report anchor as
  failed-reproduction, no claims re-anchored.
- No repo knob changed; mask conversion uses the repo's own mapping verbatim.
