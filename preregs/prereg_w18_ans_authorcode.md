# W18: ANS worst-case search on official Trident code (frozen before runs)

Closes the last audit caveat: "ANS figures are our-protocol-only". Runs the
frozen ANS protocol end-to-end on the unmodified official Trident release
(the cheapest official pipeline: ~12 s per 100-image evaluation), so the
worst-case bound is measured on author code with the authors' own forward.

## Protocol (all frozen)
- Model: official Trident, cfg_voc21.py settings, --sam_refine equivalent
  (sam_refinement=True), no repo knob changed. Vocabulary swaps recompute
  query text features exactly as the repo's __init__ does (same template
  ensemble, same normalization) -- no forward change.
- Subsets: full VOC val sorted by image id; indices 0-99 = search-100,
  100-299 = heldout-200 (disjoint; frozen here before any run).
- Candidate pools: per class, plain name + up to 5 WordNet synonyms with
  CLIP cosine in [0.70, 0.95] computed with Trident's own CLIP text tower
  (same rule as W4f); EXCLUDE list as in the original protocol.
- Search: greedy coordinate descent, one pass, alphabetical class order,
  minimize all-class mIoU on search-100; background row is never searched.
- Report: search-100 mIoU, heldout-200 mIoU for ANS / plain / syn100.

## Frozen interpretation
- The author-code ANS bound stands if heldout ANS mIoU is materially below
  the heldout syn100 value (>= 5 mIoU below); direction claim only -- no
  numeric equality with our-protocol bounds is expected or required.
- If the search fails to find anything below syn100, report as a negative
  result: the worst-case axis would then be protocol-conditional, and the
  paper caveat stays.
- Single search (alphabetical); the W16 search-order stability control is
  not re-run on author code (disclosed).
