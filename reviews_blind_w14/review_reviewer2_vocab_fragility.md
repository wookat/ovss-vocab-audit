# Review — "How Fragile Is Your Vocabulary? A Controlled Audit of Inference-Vocabulary Robustness in Training-free Open-Vocabulary Semantic Segmentation"

**Reviewer role:** Reviewer 2 (significance & novelty focus). Independent blind review; no prior reviews consulted.
**Literature check performed:** arXiv / Google Scholar / OpenReview spot-checks of the paper's positioning claims (RENOVATE, Open mIoU, Huang et al. ICLR'25, FLOSS, Neglected Tails, MESS; plus a scoop scan for controlled vocabulary-perturbation audits in OVSS). Findings incorporated below.

---

## Summary

The paper audits how sensitive training-free open-vocabulary semantic segmentation (OVSS) is to the one input the user actually controls: the inference vocabulary. Freezing a single inference protocol (OpenCLIP ViT-B/16, fixed templates, resolution, sliding window, pooling), the authors perturb only the class-name list along three axes — WordNet+CLIP-filtered synonym substitution (25/50/100%, 3 seeds), one-level hypernym granularity shift, and distractor injection (+50/+200, similarity-stratified) — across MaskCLIP, SCLIP, ClearCLIP (main matrix), NACLIP (replication), and later ProxyCLIP-/LPOSS-/SC-CLIP-style reimplementations, on VOC-21, COCO-171, ADE-150, PC-459 (~150 runs).

Main claims: (1) undocumented class-name engineering in official vocabulary files is worth 11.8–20.6 mIoU on VOC-21, comparable to or larger than method-to-method gaps; (2) synonym substitution costs 4–7 mIoU with 3.7 mIoU seed spread; hypernyms halve VOC and nearly zero ADE; (3) distractor injection *raises* GT-present mIoU by up to 10 points while dropping a classification control by 18.5 — a background-absorption pathology decomposed via a confusion-flow ledger and shown to flip sign under an engineered background or an all-class metric convention; (4) pre-registered text-side repairs (centering, vocabulary-conditioned/global ZCA whitening) improve embedding-geometry diagnostics monotonically yet reduce mIoU in 8/9 mid-to-large-vocabulary cells; (5) a constructive method (VABS: synthesized background negatives) recovers most of the engineered vocabulary's macro value (+16.1 macro on VOC-21 plain) but every pre-registered label-free mechanism for its per-class protective component fails. Extensions: ViT-L/14 direction replication, author-code ProxyCLIP anchor, a greedy adversarial name search (ANS) exposing worst cases at ~13 mIoU, cross-family transfer to grounding detectors (Grounding DINO, OWLv2, OWL-ViT v1), and a cross-lingual probe (ES/DE/RU/ZH). A large appendix "kill table" indexes ~20 pre-registered negative results. Protocol, vocabularies, and per-run JSON records are promised as a release.

---

## Strengths

**S1. The question is real, user-relevant, and — per my literature check — not yet answered in published work.** I verified the positioning: RENOVATE (NeurIPS'24) *renames* benchmarks toward better-matched names; FLOSS (ICCV'25) selects per-class *templates*; Open mIoU (arXiv 2311.03352, now TPAMI'25) and Huang et al. (ICLR'25) fix the *metric* for synonym/ambiguity and vocabulary expansion; Neglected Tails (CVPR'24) is classification-side concept-frequency. None performs a controlled, multi-axis, method-comparative perturbation audit of the inference vocabulary under a frozen dense-prediction protocol. The paper's characterization of each of these works is accurate and fair — commendably so; it neither strawmans them nor overclaims priority ("closest works study name quality [12] or template choice [14] in isolation" is a correct statement as of my check).

**S2. Experimental hygiene is well above area norms.** A single frozen protocol across all methods and conditions; pre-registered hypotheses with frozen kill criteria and a published kill table (Table 8) including invalidated criteria flagged retrospectively (the presence-gating mIoU bar); a subset-resampling noise floor (3.1 mIoU) explicitly used to gate claims; a disclosed early protocol bug with regeneration and re-runs; per-run JSON provenance. This is rare and valuable.

**S3. The distractor finding (§4.3 + Appendix C) is the paper's most genuinely novel scientific content.** The demonstration that mIoU *rises* under distractor injection exactly when background is under-modelled — with a matched classification control falling 18.5 points, a sign flip under the engineered background vocabulary, a two-ledger confusion-flow decomposition (83–88% of stolen mass from the background row, but 29–43% foreground steal), a one-scalar background-logit control that beats 50 injected names, and the demonstration that the "collapse to ~4" under all-class mIoU is a fixed-denominator convention artifact even a presence *oracle* cannot escape — is a complete mechanistic account. It also reconciles apparently contradictory prior reports (Huang et al.'s "expansion hurts"). This is dense-prediction-specific and cannot be inferred from the classification literature.

**S4. Negative results are informative, not filler.** The geometry-vs-segmentation dissociation (whitening fixes every diagnostic, hurts mIoU in 9/9 ZCA cells) is a previously unreported and practically consequential finding; the systematic failure of five mechanism families in the same direction (linear, selection, learned text adapters, learned negative queries, learned visual adapters) converts a scattered set of failures into a thesis: engineered vocabularies encode label-derived supervision that label-free text-side machinery cannot reconstruct.

**S5. Generalization work is unusually thorough for an audit paper:** fourth method replication, ViT-L scale check, three 2024-25 visually-guided reimplementations, an author-code ProxyCLIP anchor (naming effect replicates at +14.0 NEG on unmodified official code), a cross-paradigm transfer to grounding detectors with a falsification of the text-encoder-lineage hypothesis (OWL-ViT v1), and a cross-lingual probe with translation-choice oracle controls.

---

## Weaknesses

**W1. The headline finding is quantitatively new but qualitatively expected — the significance case rests on magnitude, not surprise.** That engineered name files matter was, as the paper itself concedes, "folklore": SCLIP's 26 background sub-classes are documented in its own repo, RENOVATE showed renaming changes results substantially, and OpenSeg manually curated names years ago. Reviewer-2 question: *what decision changes?* The paper's answer (§5, "Which decisions change") — rank swaps under plain/robust/worst-case regimes, τ=0.24 when ranking by naming-engineering gap — is the right kind of answer, but it arrives late, is based partly on style-reimplementations that are explicitly "not reproductions," and the two most interesting leaderboard entries (LPOSS-style, SC-CLIP-style) are unanchored to author code. The decision-relevance claim would not survive if those reimplementations are unfaithful.

**W2. Scope: three (four) final-block variants sharing one text encoder is a narrow base for the title's generality.** The paper is admirably explicit about this (§6), and §5.3–5.4 mitigate it, but the main matrix — the ~150-run controlled audit that anchors all five headline findings — is ViT-B/16 CLIP, final-block-surgery family only. The ViT-L check already shows the naming effect is *not* stable in magnitude (SCLIP NEG collapses from +20.6 to +3.5) — which cuts both ways: it validates "direction-stable" but undermines any quantitative citation of the headline 11.8–20.6 range as a property of the field. Mask-proposal-based OVSS (OVSeg/SAN/CAT-Seg family, trained but frequently compared against training-free numbers) is entirely absent, yet those methods dominate the leaderboards users actually consult.

**W3. Evaluation subsets are small and heavily reused.** Most cells are 300-image subsets; the paper's own resampling calibration puts the noise floor at 3.1 mIoU, which is *larger* than several effects discussed (centering +0.9, VABS-over-random +3.5, official +200 drop 1.7–2.9). The full-split calibration exists only for the naming axis on VOC. Several cross-method comparisons in Tables 3–5 are made at margins within or near this floor without per-cell uncertainty quantification (no CIs, single seeds on most axes other than syn50).

**W4. The granularity axis is under-analyzed and arguably mis-scored.** First-sense WordNet hypernyms with no merged-credit scoring means a model predicting a *correct* coarser concept is charged the full penalty; the paper acknowledges this and defers merged-credit to future work, but as it stands the "halves VOC / nearly zeroes ADE" numbers conflate genuine fragility with a scoring convention the paper elsewhere (distractor axis) takes great care to disentangle. The asymmetry in rigor between axes is noticeable.

**W5. Density and organization work against the contribution.** The paper reads as an audit plus ~20 appended pre-registered probes, several of which (head-level surgery decomposition, conformal wrapper, region routing — Appendix B) are only loosely coupled to the vocabulary-robustness thesis. The kill-table discipline is laudable, but the main text repeatedly interrupts its argument with probe verdicts, and key claims (C1–C8) are referenced by number without a consolidated claims table. For a benchmark/audit paper the actionable artifact — the robust-mIoU benchmark protocol of Table 6 — occupies half a page.

**W6. Minor positioning/citation issues (verified).** FLOSS is published at ICCV 2025 but cited as arXiv; Open mIoU (2311.03352) now has a TPAMI 2025 version; reference [13] (S³, ICASSP 2026) and the paper's own July 2026 dating are internally consistent but the related-work snapshot should be refreshed at submission. OpenSeg's manual name curation — cited by RENOVATE as the key precedent for name engineering — deserves a direct citation here too, since it is precisely "undocumented class-name engineering" made documented. None of these is disqualifying.

---

## Questions

**Q1.** For the style-reimplementations (LPOSS-style, SC-CLIP-style): how close are your official-vocabulary numbers to the published ones? The ProxyCLIP anchor (within 1.2) is reassuring; without analogous anchors, should the rank-swap claims of §5 be read as scoped to "our reimplementations" rather than the published methods?

**Q2.** The naming-engineering gap on ViT-L collapses for SCLIP (+3.5) but not NACLIP (+14.3). Do you have any mechanistic account (e.g., background-absorption share of the NEG per method/backbone from your confusion-flow tooling)? This heterogeneity seems more scientifically important than the direction replication it is presented as.

**Q3.** Can you provide per-cell uncertainty (e.g., bootstrap over images) for Tables 3–5, given that your own noise floor (3.1 mIoU) exceeds several reported deltas (official +200 drop, VABS-over-random on COCO-Object)?

**Q4.** For the granularity axis: what fraction of the hypernym collapse survives merged-credit scoring (crediting the coarse name for pixels of all its children)? Even a single VOC cell would tell readers whether this axis measures fragility or convention.

**Q5.** Do trained mask-proposal OVSS methods (e.g., SAN, CAT-Seg) show the same distractor sign-flip and NEG? A single VOC cell each would substantially widen the significance of the audit, since these are the methods practitioners deploy.

**Q6.** The abstract says "about 150 evaluation runs," but the appendices describe far more (7-method leaderboard, detectors, four languages, PAMR checks). What is the actual final run count, and will *all* JSON records ship?

---

## Rating

**Score: 7 / 10** (Accept — good paper; solid, unusually honest audit with one genuinely novel mechanism finding, held back from a clear accept-highlight by scope concentration in one method family, small reused subsets, and organization.)

**Confidence: 4 / 5** (I checked the positioning claims against the literature; I have not run the artifact.)

**Verdict: Accept (poster).** The distractor/background/metric decomposition and the geometry-vs-segmentation dissociation are real contributions; the audit protocol is a usable community artifact; the pre-registration discipline should be rewarded. The headline naming result is confirmatory-at-scale rather than surprising, and the generality of the quantitative claims is bounded by the narrow main matrix — hence not (yet) a strong accept.

---

## What separates this from a strong accept

1. **Anchor or replace the style-reimplementations.** Run at least LPOSS and SC-CLIP author code under their own protocols for the naming and synonym axes (as done for ProxyCLIP). All §5 leaderboard/rank-swap claims currently rest on reimplementations disclaimed as non-reproductions.
2. **Extend one or two key cells to trained mask-proposal OVSS** (SAN or CAT-Seg): naming gap + distractor sign test. If the pathology persists there, the significance of the audit roughly doubles; if it does not, that is an even better finding.
3. **Per-cell uncertainty quantification.** Bootstrap CIs over images for every table; grey out deltas within the noise floor. Several current claims (VABS-over-random, official+200 drop) are borderline against the paper's own 3.1-mIoU floor.
4. **Fix the granularity axis:** report merged-credit scoring alongside strict scoring so the hypernym numbers measure model fragility rather than metric convention — the same disentangling standard the paper sets for the distractor axis.
5. **Explain, not just report, the ViT-L NEG heterogeneity** (SCLIP +3.5 vs NACLIP +14.3) with the confusion-flow tooling. A magnitude-predictive account of *when* vocabulary engineering pays would move the paper from audit to theory.
6. **Restructure:** move the head-level decomposition, conformal, and routing probes (Appendix B) to a separate report or clearly quarantined appendix; add a consolidated claims table (C1–C8 ↔ evidence ↔ scope); give the robust-mIoU benchmark a full section with a leaderboard-ready specification.
7. **Ship the artifact before/with the camera-ready** (harness + vocabularies + all run records), since virtually every scoped claim ("archived in the run log," "listed in Appendix D") leans on it; and refresh citations (FLOSS→ICCV'25, Open mIoU→TPAMI'25, add OpenSeg).
