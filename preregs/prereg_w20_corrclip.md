# W20: CorrCLIP (ICCV 2025 Oral) audit anchor via official code (frozen before runs)

Sixth author-code anchor and first mask-generator-family (SAM2 region
masks + MetaCLIP + DINO) audit subject; recommended by the 2026 survey as
the current training-free performance ceiling representative.

## Protocol
Unmodified official CorrCLIP release (github.com/zdk258/CorrCLIP),
default config (metaclip_fullcc ViT-B-16-quickgelu, dino_vitb8,
mask_generator=None with the authors' pre-generated SAM2 region masks),
changing only the class-name files. Datasets: VOC-21 and COCO-Object
(Context-60 optional; if run, subject to the same conversion caveat as
W17-C/W19). Arms: official / plain / syn100 (frozen s0 vocabularies used
for all previous anchors).
- Reproduction gate: official arm within 1.5 of published ViT-B numbers
  (VOC-21 74.8, COCO-Object 43.7, Context-60 44.2; taken from the arXiv v3
  HTML table before any run); else failed reproduction, no interpretation
  for that dataset.
- Frozen expectations: NEG > 0 where the repo ships engineered names;
  syn100 < plain (direction only).
- Note frozen in advance: the pre-generated region masks are
  vocabulary-independent (mask proposals do not see class names), so all
  arms share identical masks; any deltas are attributable to the
  text/name interface.
Single run per arm; environment workarounds disclosed; no repo knob
changed.
