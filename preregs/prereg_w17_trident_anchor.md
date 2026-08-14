# W17: official Trident anchor -- naming axes + VABS transfer (frozen before runs)

Trigger: reviewers list Trident/TCC absence as REVA's last non-blocking
significance gap; the audit has no author-code anchor from the SAM-coupled
family. Official Trident (CVPR'25? -- use github.com/YuHengsss/Trident as
released), unmodified, its own protocol and checkpoints, full VOC val.

## Arms (only the class-name list changes)
1. official: repo's shipped VOC class names (must reproduce its published
   VOC-21 ViT-B/16 value 67.1 within 1.5 mIoU, else report
   failed-reproduction and stop); README eval command with --sam_refine;
   name files use the repo's '; ' delimiter convention;
2. plain: voc21_plain.json names;
3. syn100: voc21_syn100_s0.json names;
4. plain+VABS64: VABS negatives appended to the background entry;
5. plain+rand64: matched random negatives.

## Frozen interpretation
- Audit axes replicate if NEG = (1)-(2) > 0 and (3) < (2).
- VABS transfer holds if (4)-(2) >= +3; selection advantage if (4)-(5) > 0.
- Trident already uses SAM internally; arm (4) therefore tests whether VABS
  adds value INSIDE a SAM-based pipeline -- if (4)-(2) < +3, report as
  attenuation with the honest reading that Trident's own background handling
  may absorb it (analogue of the LPOSS absorption).
- No knob of the repo is changed; if its name format lacks a background row,
  follow the repo's own background convention and document it before running.
- Setup cost cap: if the official environment cannot be built in ~one working
  session (missing weights, incompatible CUDA), report infeasible and keep
  the papers' current "Trident comparison unrun" caveat unchanged.
