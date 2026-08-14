# Review — "How Fragile Is Your Vocabulary? A Controlled Audit of Inference-Vocabulary Robustness in Training-free OVSS"

**Reviewer 3 (practitioner perspective — training-free OVSS researcher). Independent blind review.**

My review lens: *Will this change how I run my own experiments and write my own papers? Can I actually re-run the protocol on my next method?*

---

## Summary

The paper audits inference-vocabulary robustness of training-free OVSS under a single frozen inference protocol (OpenCLIP ViT-B/16, 80 templates, fixed sliding-window/pooling), perturbing only the class-name list along three axes: WordNet+CLIP-filtered synonym substitution (25/50/100%, 3 seeds), one-level WordNet hypernym granularity shift, and distractor injection (+50/+200, near/mid strata). Main matrix: MaskCLIP/SCLIP/ClearCLIP × VOC-21/COCO-171/ADE-150/PC-459 (~150 runs), with replications on NACLIP, ViT-L/14, three 2024–25 visually-guided style-reimplementations (ProxyCLIP/LPOSS/SC-CLIP-style), an author-code ProxyCLIP anchor, and a cross-family transfer to grounding detectors (Grounding DINO, OWLv2, OWL-ViT v1) via a box→SAM harness.

Headline findings: (1) undocumented class-name engineering is worth 11.8–22.1 mIoU — comparable to or larger than method gaps; (2) synonym-seed spread reaches 3.7 mIoU, larger than many claimed improvements; (3) distractor injection *raises* GT-present mIoU via background absorption while an all-class convention collapses to ~4 — the sign of "expansion helps/hurts" is a metric-convention and background-modelling artifact, backed by a confusion-flow decomposition and a one-scalar background-logit control; (4) pre-registered text-side repairs (centering, ZCA vocab/global) improve every geometry diagnostic yet reduce mIoU in 8–9 of 9 mid-to-large cells; (5) a constructive automation attempt (VABS) recovers the macro background-sink component (+16.1 macro on VOC-21) but every pre-registered label-free mechanism for the per-class protective component fails, indexed in a kill table of ~20 pre-registered negative results. The paper proposes a robust-mIoU reporting convention (mean/worst over {plain, syn50×3, syn100}, NEG, distractor column) plus a searched worst case (ANS).

The artifact README describes a unified pipeline, per-experiment `probe_*.py` runners, pre-registration files with frozen kill criteria, a verdict log, and one JSON per run to which every paper number resolves.

## Strengths

**S1 — The protocol is genuinely reusable, and I would use it.** The audit design has the properties I need to re-run it on my own method: a frozen inference protocol shared across all conditions, frozen perturbation rules (WordNet+CLIP window [0.70, 0.95], first-sense hypernym rule, distractor guard ≤0.92 with strata), released vocabularies and fixed subset lists, and an exclusion rule for non-semantic classes. "Each `probe_*.py` is standalone" plus one-JSON-per-run provenance is exactly the right granularity. This is a rare audit paper where the protocol looks *cheaper* to adopt than to ignore.

**S2 — The robust-mIoU reporting spec is concretely executable.** Table 6 defines robust = mean, worst = min over a *named, released* vocabulary set; NEG = official − plain; distractor column with both conventions in parentheses. As a paper author I could add this table for my method in one GPU-day on the released 1349-image split. The accompanying decision-relevance analysis (Kendall τ, 8/21 pairwise flips, the official leader collapsing to 13.7 under search) answers the "so what" question most audits dodge.

**S3 — The dense-specific mechanism work is the paper's core scientific contribution.** The distractor result (classification control −18.5 vs segmentation +10; gain reverses under the engineered background vocabulary; all-class collapses to ~4) plus the confusion-flow two-ledger decomposition (83–88% of steal from background row, but 29–43% foreground steal too) and the devastating one-scalar control (a +0.02 background-logit boost beats the full 50-name gain) together settle that (a) vocabulary sensitivity cannot be imported from classification, and (b) "negative vocabulary" claims must be benchmarked against a one-parameter calibration control. I will use that control in my own work.

**S4 — The kill table is unusually valuable.** ~20 pre-registered hypotheses with frozen criteria and honest outcomes (Table 8), including three signal families each for word-space repair and presence gating, and the disclosure that an ill-posed frozen criterion (the presence-gating mIoU bar) was kept and its invalidity disclosed rather than re-judged. This is the negative-result reporting standard I wish were the norm; it will save my group months of dead-end repair attempts (I have personally considered at least three entries in this table).

**S5 — Serious triangulation against internal artifacts.** Author-code ProxyCLIP anchor (naming and synonym effects replicate on unmodified code), ViT-L/14 direction replication, subset-resampling noise floor (3.1 mIoU) explicitly stated and used, PAMR check, protocol-bug disclosure with archived-vs-corrected numbers. The self-audit discipline is state of the art.

## Weaknesses

**W1 — Main-matrix cells rest on 300-image subsets with limited seed coverage.** The subset calibration (§6) is honest, but the noise floor (3.1 mIoU) is of the same order as several tabulated effects (synonym 25%, some distractor deltas, the KC2 conditioning margin of 0.3–1.0). Most probes are single-seed. The headline effects (NEG, hypernym collapse, distractor sign flip) are far above the floor, but a practitioner copying Table 2-style numbers into a related-work comparison could easily over-read the smaller cells. The full-split Table 6 exists for VOC only.

**W2 — The style-reimplementations weaken the leaderboard claims.** LPOSS-style and SC-CLIP-style are unanchored reimplementations (disclosed), yet §5.3's most quotable finding — label propagation is the most robust mechanism, then collapses under ANS — hangs on them. The +7-point NEG discrepancy between author-code ProxyCLIP (+14.0) and the reimplementation (+22.1) shows protocol details modulate effect sizes materially. One more author-code anchor (LPOSS has public code) would substantially harden §5.3.

**W3 — Robust-mIoU spec has residual degrees of freedom that will fragment adoption.** If two future papers report "robust-mIoU" I need them comparable. Not yet fully pinned: (a) which mIoU convention is canonical for the distractor column (the paper itself shows the convention silently decides the sign); (b) whether the ANS searched-worst axis is required or optional, and whether searched vocabularies are frozen per-dataset or re-searched per method (re-search makes numbers non-comparable); (c) how to apply the protocol to methods whose official pipeline differs (PAMR, multi-scale, background thresholds — the ProxyCLIP anchor shows these interact with NEG). A one-page normative "reporting checklist" with exact file names and conventions would fix this.

**W4 — The README, as the sole artifact evidence, is below the paper's own standard.** It is a fine top-level map, but for *this* paper — whose entire value proposition is re-runnability — it leaves key practitioner questions open: no environment spec (Python/torch/OpenCLIP versions pinned? OpenCLIP QuickGELU vs GELU has bitten everyone in this subfield), no dataset-preparation instructions beyond "set the dataset roots", no expected-runtime or GPU-memory guidance for the ~150-run matrix, no lookup-script usage example showing how a specific table cell resolves to a specific JSON, no license, and no smoke test ("run this one command, expect 55.4±ε on VOC-21 test-300"). The mapping from `probe_w11_j5_prune.py`-style names to paper sections is stated to exist but not shown. I believe the artifact is complete; I cannot verify from the README alone that it is *usable in an afternoon*, which is the practitioner bar.

**W5 — Granularity axis is the least interpretable of the three.** The frozen first-sense hypernym rule admittedly produces non-visual parents, and merged-credit scoring is deferred; the near-zero ADE-150 numbers therefore conflate genuine coarse-name fragility with scoring that charges correct-but-coarse predictions as errors. As a lower bound it is fine, but I would not yet ask authors to report this axis as-is; the synonym and distractor axes carry the recommendation on their own.

**W6 — Density.** Sections 4.3–4.6 and Appendix B pack many pre-registered probes into long paragraphs; several findings that practitioners will want to cite (the one-scalar control, the contamination curve, the convention re-ranking check) are buried mid-paragraph. The kill table helps but points to prose, not numbered claims.

## Questions

**Q1.** For a new method with its own official pipeline (multi-scale, PAMR, background threshold): does the protocol demand re-implementation under your frozen harness, or is the author-code path (change only the name file, as in the ProxyCLIP anchor) an accepted mode? Which of the two produces the "official" NEG a paper should report?

**Q2.** Which mIoU convention do you propose as canonical for the reported distractor cell, given your own finding that the convention decides the sign? Would you endorse reporting the two-ledger decomposition numbers instead?

**Q3.** Is the ANS search axis intended to be re-run per method (cost? the greedy pass over WordNet candidates × classes) or do you release frozen searched vocabularies that transfer (your own transfer matrix suggests they do)? If frozen, on which method should the canonical search be run?

**Q4.** Artifact: are environment versions pinned (in particular OpenCLIP version and QuickGELU handling), and is there a single smoke-test command with an expected number? What is the total compute for reproducing the Table 6 suite for one new method?

**Q5.** The VABS dev set is 100 images disjoint from test-300, but both live inside VOC val — were the Table 6 full-split rows (1349 "dev-excluded" images) excluded of exactly those 100, and is the exclusion list released?

**Q6.** The synonym window [0.70, 0.95] and distractor guard 0.92 were frozen but not ablated. Do you have any evidence (even one cell) that mild shifts of these thresholds don't change the *ordering* conclusions, i.e. that a future suite re-generation with different thresholds remains comparable?

## Rating

**Overall: 7 / 10 — Accept.**
This is the audit our subfield needs, executed at an evidentiary standard (pre-registration, kill criteria, provenance, author-code anchor, convention checks) well above the norm. It will change practice: I intend to report the exact name file, seed-spread, and NEG for my next method. The weaknesses are real but bounded: subset/seed-limited small cells, two unanchored reimplementations under the most-quoted §5.3 finding, an under-specified reporting standard, and an artifact README that does not yet demonstrate afternoon-level usability.

**Confidence: 4 / 5.** I work directly on training-free OVSS and have reproduced several of the audited methods; I have not run the artifact (README-only evidence) and did not verify the WordNet/statistics details line by line.

**Verdict: Accept (poster; spotlight if W2–W4 are addressed in rebuttal).**

---

## What separates this from a strong accept (concrete checklist)

1. **Anchor LPOSS (and ideally SC-CLIP) on author code** for the §5.3 robustness-ranking and ANS-collapse claims, as done for ProxyCLIP — or downgrade those claims to "style-reimplementation" scope in the abstract/intro, not just §5.3.
2. **Ship a normative one-page reporting checklist**: exact vocabulary files, canonical metric convention per column, required vs optional axes (searched worst case, granularity), and the accepted author-code evaluation mode — so two independent papers reporting "robust-mIoU" are guaranteed comparable.
3. **Bring the artifact to afternoon-usability**: pinned environment, dataset-prep instructions, a one-command smoke test with expected output, the probe-name→paper-section index, per-experiment runtime estimates, a worked example of the number→JSON lookup, and a license.
4. **Multi-seed (or full-split) confirmation of the small-effect cells** that the paper itself invites readers to reuse (Table 2 syn25, Table 4 KC2 margins), or explicit "within noise floor" flags in the tables.
5. **Threshold-robustness spot-check** for the frozen suite-generation parameters (Q6), so suite regeneration doesn't silently fork the benchmark.
6. **Fix the granularity axis** with merged-credit scoring (already flagged as future work) or clearly mark it as a stress bound not to be reported in the standard table.
7. Minor: promote the one-scalar background-calibration control and the contamination curve to numbered, boxed takeaways — they are the two results practitioners will most want to cite and are currently buried.
