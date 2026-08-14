# Rebuttal Drill — REVA (final draft)
**Role:** hostile CVPR/ECCV reviewer (R2), looking for substantive reject-grade attacks (not wording).
**Basis:** final draft + full revision history (v2 text, W7–W13 deltas: 7 methods, official-ProxyCLIP anchor, J5 detector-guided pruning, (iv-b) boost control, matched official+SAM gap-closure).
*Note: the final PDF attachment failed to download (Unauthorized); this drill is grounded in the v2 full text plus all disclosed deltas since. If a specific number moved in the final PDF, re-check before submission.*

---

## Attack 1 — "The paper's own controls show a free, static hand list beats the proposed method. Why does VABS exist?"

**The attack (strongest in the stack).**
Table (controls §4.5): the hand-written background control — SCLIP's fixed 26-entry list applied to the *plain* vocabulary with the same SAM pooling — scores **59.0 / 58.4 / 53.0 / 61.0**, equal to or better than REVA (**58.7 / 56.1 / 52.7 / 59.2**) on **all four** original methods, by up to +2.3 (ClearCLIP) and +1.8 (NACLIP). The paper's stated contribution is "automation for vocabularies where no hand-engineered expansion exists" — but the control demonstrates that a *generic, vocabulary-agnostic* background list already exists, is free, transfers across methods, and wins. A user with an arbitrary plain vocabulary does not need facility-location optimization; they need to paste 26 words. On Context-60 the selection advantage over random negatives is zero, and on COCO-Object it shrinks to +0.8/+1.1 — so the "vocabulary-adaptive" part of VABS is only demonstrably useful on VOC-21, where the hand list beats it anyway. **Verdict path: the method's central automated component is dominated by a trivial baseline the authors themselves report → reject as a method paper.**

**Can the authors rebut with existing evidence? Partially — this is the weakest flank.**
- Existing: the (iv-b)/L5 control (VABS beats a *dev-tuned* scalar background boost by 2.8–4.4 despite giving the opponent labelled tuning data) shows VABS ≠ pure calibration. The selection-vs-random margin (+4.7 to +6.0 on VOC) shows adaptivity is real *relative to random*. The honesty framing ("automation, not superiority") pre-empts the tone but not the substance: "matches or slightly exceeds VABS" is in the authors' own words.
- Missing: no experiment where the fixed hand list *fails* and VABS *succeeds*. Without that, "adaptive" is untested as a differentiator.

**Cheapest rebuttal experiment.**
Run the fixed 26-entry hand list vs. VABS on the two protocols whose target vocabularies differ most from VOC: **ADE-150-style plain vocabulary and COCO-Object**, same SAM pooling, 2 methods. If VABS wins where the hand list's implicit VOC assumptions break (e.g. hand list entries collide with target classes and get filtered by τ_sim, or under-cover the scene), the adaptivity claim gets its first positive evidence. One day of compute. If the hand list still wins everywhere, the paper should be re-framed around region arbitration + *any* background sink, with VABS demoted to a tie-breaking convenience — before a reviewer forces it.

---

## Attack 2 — "Every number lives in the authors' own weakened protocol; the one method with published code shows a 7.5-point reproduction gap, and the two direct competitors are never run."

**The attack.**
All claims are within-protocol deltas in a minimal pipeline whose absolute numbers are far below published ones (NACLIP official 56.6/56.9 vs. published 64.1). Trident (ICCV'25) is a training-free CLIP+SAM competitor whose SAM-refinement functionally overlaps region arbitration; TCC is the closest neighbour to VABS. Neither is run, in either direction (no Trident row; no REVA-on-Trident). The style-reimplementations of ProxyCLIP/LPOSS/SC-CLIP are self-built; the official-code anchor validates the *base method* fragility (ProxyCLIP official 61.2, reimpl within 1.2; NEG +14.0) but — unless added in the final PDF — **plain+REVA was never executed on official code**. A hostile reviewer writes: "the gains may not survive contact with any published-strength pipeline; the missing comparison is the most obvious one; within-protocol framing is a shield, not an answer."

**Can the authors rebut with existing evidence? Mostly yes on soundness, no on significance.**
- Existing: the protocol disclosure is explicit and repeated; deltas are matched-arm; the official-ProxyCLIP anchor + ≤1.2 reimpl agreement transfers credibility to the reimpl stack; the audit paper independently shows fragility on author-released code. Killed propagation-smoothing (audit) argues PAMR-style post-processing does not repair vocabulary damage, so the delta plausibly survives stronger pipelines.
- Missing: any row proving REVA's *gain* (not just the base's fragility) exists outside the authors' stack, and any Trident/TCC number.

**Cheapest rebuttal experiment.**
One cell: **official ProxyCLIP code + plain vocabulary + VABS + SAM arbitration, VOC-21** (the base and vocab machinery already exist from the anchor run). ~1 GPU-day. A second cheap row if time allows: **NACLIP + PAMR ± REVA** to show the delta survives post-processing. Trident itself can stay future work *if* these two rows exist; without them, "no external validation of the gain" stands as written.

---

## Attack 3 — "The headline effect exists on exactly one benchmark, and the whole recipe is shaped on that benchmark."

**The attack.**
The +19.9~+23.1 / 87–90% gap-closure headline is VOC-21 only. On COCO-Object the transfer is admitted to be partially inflated by lexicon↔GT-background leakage ("overstates what a fully leak-free lexicon would achieve"); on Context-60, VABS is indistinguishable from random and the total effect is +2.0/+3.1 generic smoothing. Hyper-parameters (M=64, τ_sim=0.9) were tuned once on 100 VOC images and never re-tuned; dev is VOC; backbone is a single ViT-B/16; the full-split random control is single-seed. A reviewer assembles this into: "a VOC-shaped pipeline evaluated where its assumptions hold, with both transfer benchmarks either leaky or null — the paper generalizes a one-benchmark phenomenon into a method."

**Can the authors rebut with existing evidence? Yes, substantially — if framed as gap-tracking.**
- Existing: the audit paper shows the plain-vs-official *gap itself* is concentrated on VOC-21 (large background engineering) and small on Context-60 (59 classes cover the scene). REVA's effect tracking the size of the gap it claims to close is *predicted*, not embarrassing: zero room → zero selection advantage is the audit's own boundary condition, cited as such. The 7-method extension (three 2024–25 mechanism families, 85–89% matched-compute closure) already answers "one method family". J5's ADE-150 result (+2.5~2.7 where CLIP top-k pruning is net-negative) gives one non-VOC positive.
- Missing: a clean (leak-free) second benchmark with a *large* gap, and a second backbone.

**Cheapest rebuttal experiment.**
(a) Present a small "gap vs. recovery" table (plain→official gap and REVA recovery per benchmark) making the tracking explicit — zero new compute, converts the attack into a confirmation of the audit's model. (b) **ViT-L/14, SCLIP+NACLIP, plain/REVA/official, VOC-21** (6 cells, ~1 day) to retire the single-backbone line. (c) If a leak-free lexicon variant (remove COCO-Stuff names from L) can be run on COCO-Object in rebuttal time, it simultaneously fixes the leakage caveat and provides the missing clean transfer point.

---

## Summary table

| # | Attack | Reject-grade? | Rebuttable now? | Cheapest fix |
|---|--------|----------------|-----------------|--------------|
| 1 | Hand list dominates VABS | Yes (method-paper framing) | Partially (L5 boost control) | Hand list vs VABS on ADE/COCO vocabularies |
| 2 | No external validation of the gain; no Trident/TCC | Yes (significance) | Base yes, gain no | Official-ProxyCLIP + REVA, 1 cell; NACLIP+PAMR ± REVA |
| 3 | One-benchmark effect, VOC-shaped recipe | Borderline | Largely (gap-tracking argument) | Gap-vs-recovery table + ViT-L/14 6 cells + leak-free lexicon |

**Overall drill verdict:** Attack 1 is the one that can actually kill the paper in its current framing — it uses only the authors' own numbers and no rebuttal experiment currently answers it. Attacks 2–3 are survivable with 2–3 GPU-days of targeted cells plus framing work. Priority order for rebuttal-period compute: A1 experiment ≥ A2 ProxyCLIP cell > A3 ViT-L cells.
