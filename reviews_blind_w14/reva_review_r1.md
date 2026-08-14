# Review — Reviewer 1 (methods track, senior)

**Paper:** REVA: Region-Evidence Arbitration with Vocabulary-Adaptive Background Synthesis for Vocabulary-Robust Training-free Open-Vocabulary Segmentation

---

## Summary

The paper addresses the plain-vs-engineered vocabulary gap in training-free OVSS documented by a companion audit ([9], same author). REVA combines (i) **VABS** — facility-location selection of 64 negative "background" sub-queries from a fixed COCO-Stuff∪ADE scene lexicon, with a cosine safety filter (τ=0.90), and (ii) **region-evidence arbitration** — averaging dense per-pixel class probabilities inside SAM ViT-B automasks and re-assigning each region by pooled evidence (pixel fallback). On VOC-21 (dev-excluded 1349 images) it lifts four training-free bases from plain-vocabulary baselines by +19.9 to +23.1 mIoU, closing 86–96% of the gap to a matched-compute official+SAM upper bound, and adds +4 to +9 on top of the official vocabulary. The evidence chain is pre-registered with kill criteria; two of the authors' own stages (SLIC pooling, DINOv2 veto) are reported as killed. Extensive controls are included: matched random negatives, a dev-tuned scalar background-logit boost (self-attack), a hand-written 26-entry background list, LaVG-style feature pooling, PAMR and Trident-refinement anchors, and an unmodified-author-code ProxyCLIP anchor. A §4.6 "third component" adds OWLv2 detector-guided vocabulary pruning (+1.4 to +9.2, mean +4.4, nine cells). Transfers: COCO-Object (+12/+14, with a disclosed lexicon-leakage caveat), Context-60 (+2/+3, selection advantage vanishes), ADE-150 (VABS null, pooling +1), ViT-L/14, and three 2024–25 style-reimplementations (ProxyCLIP/LPOSS/SC-CLIP).

I verified the headline numbers against the experiment adjudication log; the paper's tables and its caveat wording (boost 73–85%, hand-list off-VOC +1.3/+2.2, matched-bound gap closures 85.2–89.4%) faithfully match the recorded runs, including several places where the log's verdicts are unflattering.

## Strengths

**S1. Exemplary experimental honesty.** The pre-registered chain with kill criteria, the reported self-kills (SLIC, DINOv2 veto), the self-attack scalar-boost control, and the matched-compute reframing of the headline (official+SAM bound, not the pixel-level official reference) are far above the field's standard. The paper does not hide that feature pooling matches probability pooling, that a hand list matches VABS on VOC, that PAMR matches SAM arbitration on the official vocabulary, or that person/tvmonitor remain harmed.

**S2. Control completeness is unusually good.** Matched random negatives (selection advantage +4.7 to +5.9 on VOC), a labelled-dev-tuned scalar background boost with a transfer check that cleanly kills it as a portable competitor (harms every transfer cell; structurally inapplicable on ADE plain), the 26-entry hand list transplanted off-VOC, a matched-budget top-k baseline for detector pruning, and same-protocol PAMR/Trident anchors. This is close to the full set of controls I would have demanded.

**S3. Breadth of validation.** Seven base methods spanning three mechanism families (attention surgery, DINO-proxy, label propagation, anomaly restoration), two backbones, four benchmarks, plus an external anchor on unmodified author code (ProxyCLIP: plain 47.3 → +VABS 56.2 vs official 61.2). The boundary-condition model (gain tracks background-modelling room: VOC ≫ COCO-Obj > Ctx-60 > ADE null) is a genuine explanatory contribution, not just a scoreboard.

**S4. The problem is real and the fix is practical.** Plain-name vocabularies are what actual users type; a training-free plug-in that recovers ~20 mIoU with zero manual vocabulary work has clear utility.

## Weaknesses

**W1. Thin methodological novelty; the paper's own controls establish that no individual component is new or superior.** SAM-region pooling of dense predictions exists in several forms (CaR, LaVG, Trident); negative/background query sets are folklore plus OVDiff/TCC; facility location is a textbook selector. The paper concedes: probability pooling ≤ feature pooling; SAM automask arbitration ≈ Trident's cheaper prompt-refinement (within 0.4–0.7) and ≈ PAMR on the official vocabulary; VABS ≤ hand list on VOC. What survives is (a) the *combination* applied to the plain-vocabulary robustness problem, (b) the vocabulary-conditioned *selection* residual (+2.8 to +4.4 over the tuned boost; +4.7 to +5.9 over random on VOC — but null on Context-60 and ADE, +1.3–1.8 i.e. below the authors' own +2 bar on LPOSS and ViT-L SCLIP). For a methods paper at this venue, "automation + robustness of a combination of known parts, with no superior mechanism claim" is a borderline contribution; the paper is candid about this, which I appreciate, but candor does not add novelty.

**W2. The "training-free" framing is strained, and §4.6 breaks it.** The core pipeline already imports SAM (supervised on 1B masks; 12× the CLIP runtime). The detector-pruning component adds OWLv2 — a grounding-supervised detector whose pretraining covers the benchmark classes; the paper itself notes this "imports external supervision rather than extending the training-free recipe" and re-imports the detector's own vocabulary sensitivity (Grounding DINO would drop 37.6 under synonyms). "Training-free" here means "no fine-tuning," but the claimed advantage over trained alternatives becomes murky when inference stacks two large supervised models per image. §4.6 is single-seed, 300-image subsets, admittedly preliminary, and its mechanism story is openly unresolved (granularity account refuted in sign; absent-mass account fails at scale). It dilutes the paper and should be compressed to a short future-work paragraph or an appendix; it currently reads as a second paper's introduction.

**W3. "Vocabulary-robust" (title claim) is demonstrated on only one axis.** REVA is evaluated on plain names. The companion audit defines synonym, distractor, and searched (ANS) axes; the log shows REVA's distractor behaviour is an open failure mode (presence gating killed twice) and REVA under synonym or ANS vocabularies is not reported at all. A method titled "vocabulary-robust" should report the full robustness suite for REVA itself — even if the numbers are unflattering (I suspect VABS does nothing for the synonym axis, which the audit shows is pure inter-class confusion). At minimum the title/abstract should scope the claim to "plain-vocabulary robustness."

**W4. Residual per-class harm is worse than the framing suggests.** The pre-registered safety criterion is REVA vs *pixel-VABS* ("arbitration adds no harm") — a lenient comparator. Against the plain baseline, person drops up to −8.9 and tvmonitor up to −11.8 on the main four, and the log shows the signature is *larger* on the newer bases (LPOSS person −15.4, tvmonitor 2.1 i.e. near-destroyed). A user cannot know in advance whether their vocabulary contains a person/tvmonitor-like class. The paper discloses this, but the abstract's "+19.9 to +23.1" headline and the "no per-class harm" safety-criterion phrasing will be read more favourably than the Table 3 reality.

**W5. Statistical fragility of several supporting claims.** Random-negative control: single seed at full-split scale. Many transfer/boundary cells: test-300 subsets, single seed. The authors' own W4a experience (a test-300 ranking headline overturned at full split) demonstrates exactly why this matters; the selection-advantage margins (+1.3 to +5.9) are of the same order as plausible subset noise. Hyper-parameters (M=64, τ=0.90) were fixed on a VOC dev split and reused everywhere — reasonable, but a sensitivity sweep is absent from the paper.

**W6. Protocol gap to published pipelines.** The unified minimal protocol (no PAMR, no multi-scale) yields NACLIP 56.9 vs published 64.1. The within-protocol-delta defense is legitimate, and the N4 check (PAMR amplifies rather than shrinks the naming gap, +15%) helps, but the full published stacks (multi-scale, official repos) remain unrun; absolute claims cannot be compared to the literature, which limits impact assessment.

**W7. Circular dependency on the companion audit.** Problem definition, protocols, the dev split, the boundary-condition model, and several negative results live in [9] (unpublished preprint, same author). The paper is not self-contained: a reader cannot verify the plain/official vocabulary construction, the 12–21 mIoU premise, or the metric-artifact claims without the companion. Key protocol details should be reproduced in an appendix.

## Questions for rebuttal

1. **Q1 (W3):** Report REVA (VABS+SAM) on the synonym (syn100) and searched (ANS) vocabularies, at least on SCLIP/NACLIP VOC. If REVA does not help those axes, revise the title/abstract scope.
2. **Q2 (W5):** Provide ≥3 seeds for the random-negative control at full split, with a confidence interval on the +4.7–5.9 selection advantage.
3. **Q3 (W4):** Can a user-facing diagnostic predict which classes will be harmed (e.g., classes visually close to selected negatives)? Even a partial predictor would materially change the adoption story.
4. **Q4 (W2):** For §4.6, what is the wall-clock and memory cost of the full CLIP+SAM+OWLv2 stack per image, and how does the *total* pipeline compare against simply running OWLv2+SAM (72.5 on VOC) where the vocabulary is detection-friendly? The dense-vs-dense-pruned scoping is stated, but the practical decision boundary is not.
5. **Q5 (W6):** One cell of REVA under a publication-grade stack (multi-scale + PAMR, any one base method) to show the +20-level gain is not an artifact of the weak protocol.
6. **Q6:** VABS negatives become sub-queries of `background`. On protocols with no background class (ADE), was an "append a background channel" variant considered rather than declaring the null? The current null may be an interface artifact.
7. **Q7 (W1):** The hand list matches VABS on VOC and loses by only +1.3/+2.2 off-VOC (single seed). Is the automation advantage worth 64 extra text queries per vocabulary vs shipping one static 26-entry list plus the SAM stage? A cost-normalized comparison would sharpen the contribution claim.

## Ratings

- **Novelty:** 2/5 — combination of known components; the paper itself demonstrates non-superiority of each part. The selection residual and boundary-condition model are the only genuinely new elements, and the former is regime-limited.
- **Technical quality / rigor:** 4.5/5 — pre-registration, kill criteria, matched controls, external anchor; docked half a point for single-seed/subset cells at claim-bearing margins.
- **Clarity:** 3/5 — dense, honest, but overloaded (the §4.6 detour, run-log prose style); not self-contained without [9].
- **Significance:** 3/5 — real practical problem, gains real; but capped by the two-supervised-models inference cost, the residual per-class harm, and the single-axis robustness scope.

**Overall: 4/10 — Borderline, leaning reject** (Weak Reject). **Confidence: 4/5.**

This is the most honest borderline paper I have reviewed in some time, and I want to be explicit that the rating reflects a novelty ceiling, not soundness: the experiments are more trustworthy than most accepted papers'. But at a methods venue, a plug-in whose every component is shown (by the authors) to be matched by a simpler alternative in its home regime, whose distinctive advantage (negative *selection*) survives only in background-rich vocabularies, and whose title claim covers one of four robustness axes, does not clear the bar as a method contribution. It might clear it as an analysis/benchmark paper with the method as a constructive corollary — see below.

## Verdict

**Weak Reject (borderline).** I would upgrade to Weak Accept given: Q1 answered (scope fixed or synonym/ANS results added), Q2 (seeded selection advantage), and §4.6 demoted. I will not fight an accept if other reviewers weight the rigor and practical utility more heavily.

## What's missing for a strong accept

1. **A mechanism, not a recipe.** Either (a) show *why* region pooling protects against negative-absorption where pixel evidence fails — beyond the oracle AUC — with a predictive account of which regions/classes are recoverable, or (b) a principled negative-selection objective that provably (or at least empirically, across regimes) dominates random+calibration, including on Context-60-like vocabularies.
2. **Full robustness suite for REVA itself** (plain, synonym, ANS, distractor, cross-lingual), so the "vocabulary-robust" title is earned; at present robustness is demonstrated only where REVA was designed to win.
3. **Elimination or prediction of per-class harm** — the person/tvmonitor problem, which worsens on 2024–25 bases; a method that trades silent −12 to −15 IoU on unpredictable classes for a macro gain is not deployable without a harm predictor.
4. **Multi-seed, full-split results for every claim-bearing margin**, especially the selection advantage and the off-VOC hand-list comparison.
5. **Publication-grade protocol validation** (multi-scale + PAMR, official repos) for at least the headline row, plus one modern backbone where dense CLIP is strong (e.g. SigLIP-family) rather than ViT-L where all baselines degrade.
6. **A resolved story for §4.6**: either developed into a validated third component (mechanism, multi-seed, cross-dataset composition table, cost accounting vs standalone detector) or removed.
7. **Self-contained protocol appendix** (vocabulary files, plain/official definitions, metric conventions) decoupling the paper from the unpublished companion audit.
8. **Cost-utility analysis**: the SAM stage is 12× the CLIP pass and PAMR gets most of the way there on engineered vocabularies at 5× less cost; a Pareto (mIoU vs s/img) figure across {none, PAMR, Trident-refine, REVA, REVA+prune} would let a reader decide when REVA is worth it.
