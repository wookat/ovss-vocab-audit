# W16: ANS search-order multi-seed control (frozen before runs)

Trigger: final audit re-review camera-ready item — the ANS vocabulary is produced
by a deterministic greedy coordinate descent with alphabetical class order; is the
worst-case bound stable if the coordinate visit order changes?

## Design
probe_ans.py extended with --order-seed: seed>=0 shuffles the coordinate visit
order with random.Random(seed) (candidate pools, search subset [0,100), heldout
[100,300), and greedy rule unchanged; the shipped run = alphabetical order).
Variant: SCLIP only (cheapest audited dense variant; the audit's ANS transfer
table already covers cross-method transfer). 2 new searches (seeds 0, 1).
Report: search-100 mIoU and heldout-200 mIoU per order, vs the shipped
alphabetical ANS (search 12.4-line values as archived; heldout_ans_200 from
w4f_ans_sclip.json), plus vocabulary overlap (n classes with identical choice).

## Frozen interpretation
- Bound stable if heldout-200 ANS mIoU across the three orders lies within
  3 mIoU: paper states the worst-case bound is search-order robust.
- Spread > 3: report the min/max as the bound range and qualify the single-order
  number.
- Either way this is a robustness check of the SEARCH PROCEDURE; no claim about
  global worst case is added.

## Amendment (disclosed before analysis of the alphabetical run)
The prereg wrongly assumed an archived alphabetical SCLIP ANS run
(w4f_ans_sclip.json); the original W4f searches were on ClearCLIP and LPOSS.
The alphabetical-order SCLIP search is therefore run fresh
(w16_ans_order_alpha_sclip.json) with identical protocol; the frozen 3-mIoU
stability criterion is unchanged and applies to the three SCLIP orders
(alpha, s0, s1).
