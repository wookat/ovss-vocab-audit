# W19: RF-CLIP (AAAI 2026) audit row via official code (frozen before runs)

User-requested leaderboard freshness: the audit benchmark tops out at 2025
methods. Intelligence survey (2026-08-02) shortlists RF-CLIP (AAAI 2026,
arXiv 2511.16170, github.com/liblacklucy/RF-CLIP) as the lowest-cost 2026
addition: CLIP-only, SCLIP/NACLIP mmseg lineage, ViT-B/16.

## Protocol
Unmodified official RF-CLIP release on the datasets it supports among
VOC-21 / Context-60 / COCO-Object, its own configs and protocol, changing
only the class-name definitions (same convention as our other author-code
anchors). Arms per dataset: official (repo names), plain, syn100 (frozen
s0 vocabularies already used for the other anchors). No distractor arm:
mmseg-style official evaluation fixes the class count to the dataset's GT
classes, so injecting extra classes is not expressible without modifying
the repo (which anchors forbid); disclosed, distractor evidence remains
in-protocol only (as for all other anchors).
- Reproduction gate: official arm must match the published ViT-B/16 mIoU
  (VOC-21 64.8, Context-60 36.4, COCO-Object 37.9; verified from the arXiv
  HTML tables before any run) within 1.5; else report failed reproduction
  and stop interpretation for that dataset.
- Frozen expectations: NEG > 0 where the repo ships engineered names;
  syn100 < plain everywhere (direction only; sizes are method/protocol
  conditional per our own findings).
- Placement: RF-CLIP enters the paper as a fifth author-code anchor and a
  2026 leaderboard freshness point; it is NOT added to the in-protocol
  reimplementation table (different protocol), and the table caption will
  say so.
Single run per arm; environment workarounds disclosed; no repo knob
changed.
