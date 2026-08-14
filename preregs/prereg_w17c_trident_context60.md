# W17-C: official Trident anchor on Context-60 (frozen before runs)

Third dataset for the Trident author-code anchor: Context-60 is the
mid-headroom point of the in-stack dataset gradient (VOC +20ish, COCO +3ish,
Context +2-3 in-stack VABS gain), so it probes the middle of the
gap-vs-recovery boundary on author code. Unmodified official Trident,
cfg_context60.py (--sam_refine), full Context val (its own protocol), only
the class-name file changes ('; ' convention):
1. official: repo's cls_context60.txt (must reproduce published ViT-B/16
   Context60 38.6 within 1.5 mIoU, else failed-reproduction, stop);
2. plain: ctx60_plain names;
3. syn100: ctx60_syn100_s0 names;
4. plain+VABS64: ctx60 VABS negatives appended to background;
5. plain+rand64: ctx60 matched random negatives.

## Frozen interpretation
- Audit axes replicate if NEG = (1)-(2) > 0 and (3) < (2).
- VABS transfer holds if (4)-(2) >= +3; selection advantage if (4)-(5) > 0.
- Boundary prediction (stated before running): Context is expected to be a
  low-to-mid headroom regime; if NEG is small, VABS ~ random (selection
  NULL) is the predicted outcome, as on COCO-Object. Either outcome is
  reported as-is; no reclassification after seeing results.
- No repo knob changed; single run, single negative seed (disclose).
