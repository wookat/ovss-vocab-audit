# Review — Reviewer 1 (NeurIPS D&B / TPAMI standard)

**Paper:** *How Fragile Is Your Vocabulary? A Controlled Audit of Inference-Vocabulary Robustness in Training-free Open-Vocabulary Semantic Segmentation*
**Materials checked:** main.pdf; artifact verdict log (RESULTS.md, attachment 7f23e07e); artifact README (81270f99). All table/claim cross-checks below were performed independently against the paper's own tables and the run log.

---

## Summary

The paper audits training-free OVSS methods by freezing a single inference protocol and perturbing only the inference vocabulary along three axes (synonym substitution, granularity/hypernym shift, distractor injection), across 3 main methods + 1 replication (NACLIP) + 3 style-reimplemented 2024–25 methods + 2–3 grounding detectors, on VOC-21 / COCO-171 / ADE-150 / PC-459 (plus COCO-Object, Context-60, a cross-lingual probe, and a ViT-L/14 scale check). Main findings: (1) undocumented class-name engineering is worth 11.8–20.6 mIoU on VOC-21 (up to 22.1 in the extended leaderboard), comparable to or larger than method gaps; (2) synonym substitution costs 4–7 mIoU with 3.7 mIoU seed spread; hypernyms halve VOC and nearly zero ADE; (3) distractor injection *raises* GT-present mIoU via background absorption while an object-crop classification control drops 18.5 pts, and the sign of the effect is a function of background modelling and metric convention (a confusion-flow decomposition and a fixed-vs-NaN-denominator analysis make this quantitative); (4) pre-registered text-side repairs (centering, ZCA whitening, learned adapters, RECAL, word-space rewrites, LLM canonicalisation) all fail with a kill-table appendix; (5) a constructive method (VABS) recovers most of the macro value of engineered vocabularies but its per-class protective component resists every label-free mechanism; (6) a searched worst case (ANS) is 15–17 mIoU below the worst random suite member and transfers across methods and even across training paradigms, decomposed into ~75–80% non-canonical-name sensitivity plus a 5–7 pt adversarial residual.

## Strengths

- **S1 (major).** Genuinely novel benchmark contribution: no prior work performs a controlled, multi-axis, dense-prediction-specific vocabulary perturbation audit. Related-work positioning against RENOVATE, FLOSS, Open-mIoU is precise and fair (§2).
- **S2 (major).** Unusual methodological hygiene: pre-registered kill criteria, a full negative-results kill table (Table 8), disclosed criterion mis-specifications kept rather than re-judged (App. B presence-gating note), disclosed protocol bug and re-runs (App. D). The verdict log corroborates this discipline throughout.
- **S3 (major).** The distractor analysis (§4.3 + App. C) is the strongest scientific content: mechanism (background absorption) is established with converging controls — engineered-background reversal, all-class metric collapse, pixel-flow ledger (83–88% of steal from the background row, but 29–43% foreground steal honestly reported), the GT-presence oracle showing the ∼4 all-class figure is a denominator convention, and the one-scalar background-boost control. This decisively separates convention artifact from real behaviour.
- **S4.** Numbers I could re-derive are consistent: Table 1 deltas (20.6/18.7/11.8), seed means/stds (32.1±2.0 etc. match Table 2 seeds), Table 3 drops (−1.7 to −2.9), Table 5 macro gains (+16.1, random +12.6, VABS-vs-random +3.5, official +0.1–0.6, COCO macro +8.9), Table 6 matches the W4a log exactly, §5.2/§5.4/§5.5 figures match W1a/W6-F2/W7b/W8/W10/W11-J3 logs. External author-code anchor (ProxyCLIP 61.2/47.3/41.2) matches W4g.
- **S5.** The full-split remediation (Table 6 on 1349 images after the test-300 leader turned out to be subset noise) is reported *as a finding* rather than hidden — exemplary.
- **S6.** Cross-family (§5.4) and cross-lingual (§5.5) extensions, each with falsification of the obvious hypothesis (lineage banding killed by OWL-ViT v1; Spanish deficit reduced to translation-choice by the best-of-3 oracle), materially broaden the claim scope.

## Weaknesses

- **W1 (moderate, soundness — internal inconsistency).** Figure 4's caption states repairs change PC-459 3-method-mean mIoU "by at most +0.1 and down to −2.2". From Table 4 I compute: center +0.17 (12.73 vs 12.57), ZCA-vocab −1.57, ZCA-global −1.23; the largest *single-cell* PC-459 drop is −1.4. Neither +0.1 nor −2.2 is derivable from Table 4. Evidence: Fig. 4 caption vs Table 4, p.8.
- **W2 (moderate, completeness/self-consistency).** §4.3 (correctly) demands that "any absorption or negative-vocabulary claim should be benchmarked against this one-parameter [background-boost] control", yet §4.5's headline VABS result (Table 5) is *not* benchmarked against it. The artifact log (W13-L5/L5t) shows a dev-tuned scalar boost reproduces 73–85% of VABS's text-side gain on VOC (50.7 vs 53.5 on SCLIP) while failing to transfer cross-dataset — evidence that would actually *strengthen* VABS but is absent from the paper. As written, §4.5 fails the paper's own stated baseline standard. Evidence: §4.3 p.7 vs Table 5/§4.5; RESULTS.md W13-L5, W13-L5t.
- **W3 (minor–moderate, presentation/soundness).** The abstract and contribution 2 state the naming effect is "worth up to 20.6 mIoU (11.8–20.6 across methods)", but the paper's own extended leaderboard (Table 6) shows NEG up to 22.1 (ProxyCLIP-style) and the full-split calibration gives 21.7 (SCLIP, §6). The headline range is silently scoped to the 3-method test-300 matrix; the abstract should quote the wider measured range or state the scope. Evidence: Abstract; Table 6 NEG column; §6.
- **W4 (moderate, soundness — subset heterogeneity).** Numbers from different image subsets are compared across sections without a consolidated map: main matrix on test-300, leaderboard on the 1349 dev-excluded split, ANS on search-100/heldout-200, VABS dev-100/test-300, App. C on test-300. E.g. §5.3 compares ANS held-out 13.4/13.7 against "worst random-suite member 28.3/31.0" (test-300 values) in a paragraph anchored to Table 6 (full split, worst 29.8/32.7). Each individual number is disclosed somewhere, but the reader cannot always tell which split backs which comparison. Evidence: §5.3 vs Table 6; RESULTS.md W3b vs W4a vs W4f.
- **W5 (minor, soundness).** Two different OWLv2 VOC plain values are used within §5.5 without explanation: retention is computed against 72.5 ("72.5→57.3, 79%") but the best-of-3 oracle against 73.4 ("62.0 vs 73.4, 84%"). Presumably test-300 vs heldout-200, but this is not stated and changes the retention figure by ~1pt. Evidence: §5.5; RESULTS.md W7c/W8 vs W10.
- **W6 (moderate, external validity — disclosed).** Reproduction gap: SCLIP official 56.9 (full split) vs published 59.1; the PAMR check (§6) bounds only one post-processing axis (and shows PAMR *amplifies* NEG by 15%, which actually helps the thesis), but multi-scale/full published stacks remain unbounded, and the LPOSS-/SC-CLIP-style rows are unanchored reimplementations — yet leaderboard-level claims ("LPOSS-style is the most vocabulary-robust method on every aggregate") rest on them. Evidence: §5.3, §6, RESULTS.md W4g note.
- **W7 (minor–moderate, statistics).** No confidence intervals or significance tests anywhere; the noise floor is a single subset-resampling calibration (3.1 mIoU, 2 methods). Several quantitative claims sit near this floor (VABS-vs-random +3.5; LPOSS-vs-random advantage +1.3 on the LPOSS REVA row in the log; "+0.9 centering exception"). Distractor pools, ANS search, and all detector cells are single-seed (disclosed in README but should be in each caption).
- **W8 (minor, disclosed).** The classification control in Table 3 is a single cell (one encoder, one condition) and uses GT-box crops — the paper's own caveat (§4.3, §6) is fair, but a claim as headline-worthy as "dense sensitivity cannot be read off classification behaviour" (abstract, finding 3) deserves more than n=1.
- **W9 (minor).** §4.5's per-class safety ranges ("person −4.8 to −8.2 on three methods") omit the substantially larger harms on the newer bases recorded in the log (LPOSS person −15.4, tvmonitor →2.1; W4h). Since §4.5's safety conclusion is method-general, the wider range should be reported.
- **W10 (trivial).** "About 150 evaluation runs" (abstract) is hard to reconcile with the log (49+70+42+… run files across W3–W14); presumably it scopes the original audit matrix only.

## Questions

1. Please recompute or explain the Figure 4 caption values (+0.1 / −2.2) against Table 4 (W1).
2. Why is the scalar background-boost baseline (and its NOT-TRANSFER result) absent from §4.5/Table 5 when §4.3 declares it the mandatory control for absorption claims (W2)?
3. Which subset backs each comparison in §5.3 (ANS vs random worst)? Consider a per-claim subset table (W4).
4. What explains the 72.5 vs 73.4 OWLv2 plain values in §5.5 (W5)?
5. Are the distractor pools and the +50/+200 draws single-seed? If so, what is the pool-resampling variance (the archived-vs-regenerated 45.4→44.5 shift in App. D suggests ~1 mIoU)?
6. Was the Kendall τ analysis in §5 (τ=0.81 / 0.24; 8/21 pair flips) computed on the full split of Table 6? Please state.
7. Abstract says "up to 20.6"; Table 6 says up to 22.1. Which is the claim (W3)?

## Consistency audit (claims vs tables)

Verified consistent: Table 1 deltas; §4.1 ordering claims (1.8, 2.5, 0.1-tie); Table 2 seed means/stds and 3.7 spread; hypernym 13.1/14.7/10.2→2.5/3.8/1.9; Table 3 (+7.5/+9.7 gains, 18.5 cls drop, −1.7..−2.9 official reversal); Table 5 (all macros re-derived); Table 6 = W4a log; §5.1 NACLIP cells = log; §5.2 ViT-L cells = W1a; §5.3 ANS = W4f/W7a; §5.4 = W6-F2/W7b/W10; §5.5 = W7c/W8/W10/W11-J3; App. C ranges = W5e; Table 8 outcomes = log verdicts. Discrepancies found: W1, W3, W5 above, plus the NEG rounding (log 20.7 vs paper 20.6, benign).

## Rating

- **Score: 7 / 10** (Accept)
- **Confidence: 4 / 5**
- **Verdict: Accept**

Rationale: an original, decision-relevant benchmark/audit with exceptional negative-result and pre-registration discipline; the identified issues are consistency/presentation and single-seed/statistics gaps, not flaws in the central mechanisms, which are established with converging controls. It is held below Strong Accept by W1–W7.

---

## 距离强接收（Strong Accept）还差什么 — 按成本排序

1. **零成本（文字修订）**：修复 Fig.4 caption 数字（W1）；统一摘要 NEG 区间与 Table 6/全量拆分（W3）；解释/统一 OWLv2 72.5 vs 73.4（W5）；在每个表格 caption 注明 subset 与 seed 数；"约150次运行"改为准确数字。
2. **近零成本（结果已在 artifact 中，只需搬进论文）**：把 W13-L5/L5t 的标量背景 boost 基线行（VABS-DISTINCT + NOT-TRANSFER）加入 §4.5/Table 5，满足论文自设的对照标准（W2）；把 W4h 新方法 per-class 安全数字并入 §4.5（W9）；给 §5.3 的每个对比补 subset 映射表（W4）。
3. **低成本计算（数天）**：distractor 池与 ANS 搜索各补 2–3 个 seed，报告 pool-resampling 方差；分类对照从单 cell 扩到 3 方法 × 2 条件（W7、W8）。
4. **中成本**：主矩阵（Table 1–3）在全量 dev-excluded split 上重跑并给出 bootstrap CI/显著性检验，使"seed spread > 报告增益"这类比较有统计支撑（W7）。
5. **中成本**：为 LPOSS/SC-CLIP 补 author-code 外部锚点（复刻 W4g 流程），使"LPOSS-style 最鲁棒"的排行榜结论脱离 style-reimpl 限定（W6）。
6. **中高成本**：补 A-847 大词表 regime（论文自认缺口）；至少一个 multi-scale/完整发布 stack 的效应量校验，闭合 §6 承认的"unbounded"轴。
7. **高成本**：hypernym 映射的人工校验子集 + 小规模用户词表调查（把"user-plausible"从假设变为测量）；跨语言从 3 语言 k=3 扩到类型学覆盖；native segmenter（非 box→SAM）上的跨范式复验。
