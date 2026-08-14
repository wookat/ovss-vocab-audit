# Pre-registration W13-L1: metric-convention go/no-go — does the accounting convention change scientific conclusions? (frozen before any run)

Date frozen: 2026-08-01, before any recomputation.

## Question
K3 capstone + K2 contamination curves + BA decomposition suggest the
mIoU accounting convention (denominator over all vocabulary classes vs
NaN-exclusion of 0/0 classes vs GT-present-only) is an unacknowledged
free variable. Does the choice of convention change method rankings or
conclusion-level findings on our benchmark, beyond absolute values?

## Frozen design
Recompute from archived confusion matrices (no new inference where
caches exist; new inference only where per-class inter/union not
archived) three conventions:
- C-fixed: mean IoU over all K vocabulary classes (0/0 counts 0);
- C-nan: mean IoU excluding 0/0 classes (mmseg-style NaN exclusion);
- C-present: mean IoU over GT-present classes only (per dataset union).
Grid: 7 dense methods x {plain, syn100_s0, dis_near200} x VOC (full
archived splits where available, else test-300).

## Criteria (frozen)
- GO (protocol paper leg): for at least one perturbation axis, some
  convention pair has method-ranking Spearman < 0.85, OR at least 2
  conclusion-level flips (a method pair ordering that reverses, or an
  axis verdict that changes sign). The presence-gating retrospective
  mis-kill counts as 1 pre-existing conclusion flip (already
  established, not recounted from this run).
- NO-GO: all convention pairs Spearman > 0.95 on every axis and no new
  flip -> conventions move absolute values only; L1 is demoted to an
  audit-paper appendix note and no protocol paper is pursued.
- MIXED otherwise.

## Scope
One dataset family (VOC) at go/no-go stage; GO requires Context-60
replication before any protocol-paper claim. Literature-forensics leg
(which conventions published papers use) is separate and not gated by
this pre-registration.
