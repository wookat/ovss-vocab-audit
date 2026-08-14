# Stage 12 (month-long iteration) results

## W1a ViT-L/14 backbone generalization (prereg_w1_vitl.md) — MIXED, directions replicate
VOC-21 test-300, ViT-L-14-quickgelu openai, unified protocol. (GT-present / all-class %)

| method | plain | official | syn100 | plain+200near (GTp) | plain_vabs64 |
|---|---|---|---|---|---|
| SCLIP  | 37.24 | 40.72 | 30.23 | 36.09 (all 3.43) | 41.19 |
| NACLIP | 36.31 | 50.60 | 30.65 | 41.87 (all 3.98) | 47.47 |

SAM arm (SCLIP, test-300): pix_vabs 41.19, sam_reg_vabs 44.01, sam_reg_rand 42.24.

Criteria: R1 official gap: NACLIP +14.3 PASS, SCLIP +3.5 FAIL (<8).
R2 synonym drop: both PASS (−7.0 / −5.7). R3 distractor sink: PASS both
(GT-present flat or rising, all-class collapse = same metric artifact).
R4 VABS: NACLIP +11.2 PASS, SCLIP +4.0 FAIL. R5 SAM: pooling +2.8 PASS,
selection vs random +1.77 marginal FAIL (<2).

Interpretation (honest): all audit/REVA effect DIRECTIONS replicate on ViT-L/14,
but magnitudes are method-heterogeneous — SCLIP's correlative attention benefits
far less from vocabulary engineering on ViT-L, and ViT-L dense segmentation is
overall weaker than ViT-B (known). Goes into both papers as a scale-generalization
section with the SCLIP caveat, not as a uniform replication claim.

## W1b few-label per-region router (prereg_w1_router.md) — KILLED
Pool (top-4 on train-50): kk_L12 / qq_L12 / ident_L12 / qq_L11.
Ridge router on 50 labelled images (region margin/conf/entropy/size/agreement +
config one-hot):
- held-out dev-50: routed 53.42 vs best single kk_L12 54.04 (NEGATIVE transfer)
- test-300: routed 55.113 vs kk_L12 55.113 (+0.0004, router degenerates to
  always picking kk_L12)
Negative-transfer guard triggered on held-dev -> killed per prereg (no redesign,
window required gain in [0,2) on both sets). Conclusion strengthens C6r2: even
50-image supervision over margin/confidence/agreement features cannot access the
+14.9 per-region oracle; the routing signal is not in these region statistics.
Files: /media/dell/DATA/ovss/runs/w1b_router.json, vitl_*.json.

## W2a ADE-150 transfer (prereg_w2a_ade.md) — E1 marginal pass, VABS null (as predicted)
ADE-150 val first-300, ViT-B/16. Plain refs: SCLIP 13.08 (500-img run),
NACLIP 15.39 (same 300 imgs).

| method | pix_vabs | sam_reg_vabs | sam_reg_rand |
|---|---|---|---|
| SCLIP  | 13.13 | 14.17 | 14.35 |
| NACLIP | 14.74 | 16.06 | 16.16 |

- E1 SAM pooling: +1.04 / +1.32 over pixel — PASS (>=+1), region evidence
  transfers to a 150-class no-background regime, though much smaller than VOC.
- E2 VABS: no gain over plain (SCLIP +0.05, NACLIP -0.65) and equal to random
  negatives — consistent with the Context-60 finding: without a background class
  to absorb into, vocabulary-adaptive negatives have no room. Goes into REVA
  limitations as the no-background regime boundary (harm < 2, no kill).
- Disclosed: VABS lexicon contains ADE class names (tau filter applied); the
  meter includes the appended background prediction class (GT never contains it).
Files: runs/w2a_ade_{sclip,naclip}_sam.json, vocabs ade150_plain_{vabs,randneg}64.

## W2c VocabUQ go/no-go (prereg_w2c_vocabuq.md) — KILLED (K1+K2 on VOC)
SCLIP ViT-B, SAM regions; COCO-Object 500-region calibration, alpha=0.1.

| test set | coverage@90% | mean set size | frac of vocab |
|---|---:|---:|---:|
| VOC-21 (300 reg)     | 59.7% | 15.2 | 72.3% |
| Context-60 (300 reg) | 92.3% | 22.3 | 37.1% |

- K1 kill: VOC set size 72% of vocab (>50%). K2 kill: VOC coverage 59.7%
  (>5pp deviation). Ctx60 alone would pass both (92.3%, 37%).
- Diagnosis: nonconformity scores are NOT comparable across vocabulary sizes
  (softmax mass spreads over 81 vs 21 vs 60 classes), so COCO-calibrated
  thresholds do not transfer to VOC. Per-variant single-vocab thresholds were
  trivially loose on VOC (coverage 100%). A redesign with per-dataset
  calibration or size-normalized scores was NOT permitted by the frozen
  protocol; direction killed in its cross-dataset-guarantee form. Within-
  distribution form (ctx60) remains open but is a much weaker claim.
- File: runs/w2c_vocabuq.json.

## W2d SelfRoute go/no-go (prereg_w2d_selfroute.md) — KILLED (K2 both, K1 ctx)
8-config pool (4 flavors x L11/12), VOC 300-499 and Context-60 0-199, plain.

| dataset | best single | consist | consist+dino | margin ctl | oracle(8) |
|---|---:|---:|---:|---:|---:|
| VOC   | kk_L12 54.23 | 46.81 | 46.28 | 42.52 | 56.64 |
| ctx60 | kk_L12 26.41 | 21.87 | 21.66 | 22.15 | 28.64 |

- K2 kill: consistency arms < best single on BOTH datasets (need >= +2).
- K1: match rate 24.7% (VOC, marginal pass) / 16.9% (ctx, kill vs 22.5% bar).
- Margin control ~= consistency on ctx => the "changed supervision form"
  hypothesis is dead: consistency signals are no better than margin at region
  granularity. Third signal family (after margin C6r2 and few-label W1b) that
  fails to access the routing oracle. Note the 8-pool oracle ceiling is only
  +2.4/+2.2 (vs +14.9 for the 20-pool): most of the C6 headroom lives in the
  L8-10 configs that are individually weak, which no realistic gate selects.
- Files: runs/w2d_selfroute_{voc,ctx}.json.

## W2e Robust-mIoU go/no-go (prereg_w2e_neg.md) — GO
ProxyCLIP-style reimpl (DINO ViT-S/16 proxy attention over CLIP-B/16 values,
unified protocol), VOC-21 test-300:

| vocab | mIoU |
|---|---:|
| plain | 37.80 |
| official | 58.78 |
| syn100_s0 | 30.78 |

- NEG = +21.0 (>=3 GO bar), synonym drop = 7.0 (>=3): the 2024 DINO-guided
  generation is exactly as naming-sensitive as the 2022-23 attention-surgery
  family — vocabulary robustness is NOT fixed by visual guidance.
- Sanity gate: official 58.8 vs published VOC-21 ~61 (ViT-B/8 DINO, protocol
  differences) — within the 15-mIoU validity bar.
- Next: add 2 more 2024-25 baselines + robust-mIoU definition prereg; also
  test whether REVA (VABS+SAM) recovers ProxyCLIP's plain gap (method-paper
  strengthening).
- Files: runs/w2e_proxy_{plain,official,syn100}.json, proxyclip_seg.py.

## W3a REVA x ProxyCLIP — strong pass (5th method, near-full gap closure)
VOC-21 test-300, ProxyCLIP-style base (same as W2e):

| arm | mIoU |
|---|---:|
| plain (pixel) | 37.80 |
| pix VABS | 55.50 |
| REVA (VABS + SAM regions) | 58.53 |
| SAM + random negatives | 54.36 |
| official (pixel, reference) | 58.78 |

- REVA closes 99% of the naming-engineering gap (58.53 vs 58.78) on a 2024
  DINO-guided method whose mechanism differs from the attention-surgery
  family — strongest transfer evidence yet for the method paper (5th base).
- VABS selection advantage vs random negatives: +4.2 (consistent with ViT-B
  results, unlike the ViT-L SCLIP arm).
- File: runs/w3a_proxy_reva.json.

## W3b Robust-mIoU benchmark round 1 (prereg_w3b_robustbench.md) — GO
7 methods x 7 vocabs, VOC-21 test-300, unified protocol. New rows are
style-reimplementations: ProxyCLIP (2408.04883), LPOSS (2503.19777),
SC-CLIP (2411.15869).

| method | official | plain | robust* | worst-case | NEG | dis_near200 all-class |
|---|---:|---:|---:|---:|---:|---:|
| MaskCLIP  | 43.55 | 31.76 | 28.91 | 26.29 | 11.8 | 3.5 |
| SCLIP     | 55.42 | 34.75 | 32.34 | 30.55 | 20.7 | 4.2 |
| ClearCLIP | 53.58 | 34.89 | 32.24 | 29.03 | 18.7 | 3.9 |
| NACLIP    | 55.02 | 36.54 | 33.37 | 29.67 | 18.5 | 4.0 |
| ProxyCLIP | 58.78 | 37.80 | 34.51 | 30.78 | 21.0 | 4.3 |
| LPOSS     | 58.37 | 41.19 | 37.11 | 31.85 | 17.2 | 4.4 |
| SC-CLIP   | 56.66 | 38.05 | 34.68 | 30.68 | 18.6 | 4.2 |

*robust = mean{plain, syn50 s0-2, syn100 s0}; worst-case = min of same set.

- GO(a): all 3 new-generation rows NEG 17-21 (>=3 bar by 6x) — naming
  engineering is NOT fixed by DINO guidance, label propagation, or anomaly
  restoration.
- GO(b): rank swap at the top — official ranks ProxyCLIP #1, worst-case
  ranks LPOSS #1 (tau < 1). LPOSS-style is the most vocabulary-robust
  method tested (propagation smooths naming noise), a nontrivial finding.
- Distractor background sink is universal: all 7 methods collapse to
  all-class ~4 under +200 near distractors (GT-present stays 37-46).
- Files: runs/w3b_{method}_{vocab}.json (49), newgen_seg.py, proxyclip_seg.py.

## W3d Benchmark round 2: Context-60 / COCO-Object + REVA repair rows
7 methods x {plain, syn50 s0-2, syn100 s0}, test-300 each (no official vocab
exists for these datasets, so this round measures perturbation robustness only).

robust* / worst-case (mean/min over the 5-vocab suite):
- ctx60: ProxyCLIP 25.08/20.58 #1; ClearCLIP 23.52, SC-CLIP 23.49, LPOSS 23.40,
  NACLIP 23.10, SCLIP 22.09, MaskCLIP 16.90. Synonym drops 4-9 mIoU persist.
- cocoobj: LPOSS 22.70/19.56 #1; ProxyCLIP 22.11, SC-CLIP 21.80, ClearCLIP
  21.46, NACLIP 21.18, SCLIP 20.71, MaskCLIP 17.39. Drops 2-4 mIoU.
- Rank swaps vs plain ranking are minor on these datasets (LPOSS's smoothing
  advantage is VOC-specific in magnitude); the headline reshuffle remains the
  VOC worst-case swap (W3b).

REVA rows on the two new bases (VOC-21 test-300):
- LPOSS: plain 41.19 -> pixVABS 55.62 -> REVA 57.43 (94% of the 17.2 gap to
  official 58.37); VABS-vs-random advantage only +1.3 (56.13 random) — below
  the +2 replication bar, honest scoping: propagation partially absorbs the
  negative-vocabulary distinction.
- SC-CLIP: plain 38.05 -> REVA 58.49, above its official pixel reference
  (56.66); VABS advantage +3.1.
- Total REVA base methods now: 7 (4 surgery + ProxyCLIP + LPOSS + SC-CLIP),
  gap closure 86-103%.
- Files: runs/w3d_{ds}_{m}_{v}.json (70), runs/w3d_reva_voc21_{lposs,scclip}.json.

## W4a Robust-mIoU benchmark at full dev-excluded split (reviewer remediation)
7 methods x 7 vocabs, VOC-21 full 1349 (dev-100 excluded), replacing test-300:

| method | official | plain | robust* | worst | NEG | dis all(GT) |
|---|---:|---:|---:|---:|---:|---|
| MaskCLIP | 44.61 | 32.78 | 29.65 | 26.79 | 11.8 | 3.5 (37.1) |
| SCLIP | 57.21 | 35.60 | 33.49 | 31.79 | 21.6 | 4.3 (44.9) |
| ClearCLIP | 55.68 | 34.75 | 32.53 | 29.81 | 20.9 | 3.9 (41.2) |
| NACLIP | 56.93 | 36.93 | 34.03 | 30.66 | 20.0 | 4.1 (42.7) |
| ProxyCLIP-style | 60.03 | 37.96 | 35.04 | 31.74 | 22.1 | 4.3 (44.8) |
| LPOSS-style | 60.45 | 40.83 | 37.33 | 32.70 | 19.6 | 4.3 (45.5) |
| SC-CLIP-style | 58.42 | 38.28 | 35.24 | 31.61 | 20.1 | 4.2 (44.1) |

- All W3b conclusions survive EXCEPT the ranking-swap headline: at full split
  LPOSS-style tops BOTH the official and the worst-case ranking (60.45 vs
  60.03; 32.70 vs 31.74). The test-300 official #1 (ProxyCLIP) was subset
  noise, exactly as the delta reviewers warned. Honest revision: the stable
  findings are (i) NEG 11.8-22.1 universal, (ii) distractor collapse to ~4
  universal, (iii) LPOSS-style is the most vocabulary-robust method on every
  aggregate (plain, robust*, worst-case), (iv) the official-vs-worst-case
  top-1 identity is subset-sensitive, which itself argues for robust
  reporting.
- Files: runs/w4a_voc21full_{m}_{v}.json (49).

## W4c Propagation-smoothing mechanism probe: BOTH hypotheses KILLED (prereg_w4c_propsmooth.md)
- E1 spectral: vocabulary-perturbation logit noise is 97% LOW-frequency on the
  DINO graph (high-band energy fraction 0.029 vs >=0.60 required). Propagation
  attenuates the high band 12.9x more, but there is almost nothing there to
  filter. KILL H1.
- E2 transplant (6 methods, plain vs syn100, test-300): post-hoc propagation
  RAISES absolute mIoU everywhere (plain +1.7..+12.8; MaskCLIP 31.8->44.6) but
  INCREASES the synonym drop on all six methods (mean drop change -38.5%,
  required >=+25% reduction). KILL H2.
- Honest mechanism revision: LPOSS-style robustness is a higher-plain-baseline
  effect, not text-noise filtering. Naming noise is spatially class-coherent
  (graph-low-frequency), so no spatial/graph smoothing can remove it — a new
  link in the "text-side noise is not repairable downstream" chain; goes to
  audit paper appendix. Files: runs/w4c_propspec.json, runs/w4c_prop_*.json.

## W4d Presence-gated REVA: KILLED (prereg_w4d_presence.md)
ClearCLIP+NACLIP, VOC test-300, frozen tau=1.0 (median rank gate), topK=3:
- E1 distractor: all-class mIoU 3.8/4.2 gated vs 3.9/4.0 ungated pixel —
  required >=20, kill <12. KILL. The median-rank gate keeps ~half the
  221-item vocabulary (precision 0.24 at recall 0.99): region-pooled softmax
  mass under 200 near-distractors is spread so evenly that presence is not
  decidable from this score family. Same signal weakness as the three dead
  routing-signal families.
- E2 no-harm passed trivially (gate changes <=0.15 on official/plain), which
  confirms the gate is near-inert rather than safe-and-selective.
- Honest note: per prereg we do not tune tau past the frozen value; negative
  result recorded. Distractor collapse remains an open problem for REVA.
- Files: runs/w4d_{clearclip,naclip}_{dis_near200,official,plain}.json.

## W4e Matched official+SAM upper bounds for the three new bases (test-300)
- ProxyCLIP 62.09, LPOSS 60.25, SC-CLIP 60.92 (pixel official refs: 58.78 /
  58.37 / 56.66). Recomputed REVA gap closures vs the MATCHED bound:
  ProxyCLIP 85.4%, LPOSS 85.2%, SC-CLIP 89.4% (was 99/94/"above official"
  vs pixel refs). SC-CLIP REVA (58.49) exceeds its pixel official but stays
  2.4 below the matched official+SAM bound — papers updated to the matched
  framing. Files: runs/w4e_offsam_*.json.

## W4f ANS adversarial naming search: GO, strongest fragility result so far
(prereg_w4f_ans.md; ClearCLIP + LPOSS-style, VOC, search-100/heldout-200 disjoint)
- Greedy per-class synonym search (WordNet pool, CLIP cosine 0.70-0.95, one
  pass, 120 evals) finds lexicon-legal vocabularies at held-out mIoU 13.4
  (ClearCLIP) / 13.7 (LPOSS) vs 28.3 / 31.0 for the worst random-suite member
  and 34.5 / 40.8 plain — 15-17 mIoU BELOW the random worst (GO line 3.0).
  No search overfit (search 12.9/14.2 ~= held-out).
- Even the most vocabulary-robust method (LPOSS) collapses identically; its
  robustness advantage vanishes under search (13.7 vs 13.4).
- Chosen names are dictionary-sanctioned but often low-frequency senses
  (bicycle->wheel, person->soul, dog->frump): honest framing = "lexicon-legal
  worst case", an upper-bound stress instrument, not a typical-user scenario.
- Implication for the benchmark: random synonym suites understate worst-case
  fragility by ~2x; robust-mIoU gains a searched worst-case axis.
- Files: runs/w4f_ans_{clearclip,lposs}.json (incl. full vocabularies+trace).
- Transfer matrix (reviewer remediation): each method's searched vocabulary is
  nearly as damaging on the OTHER method — lposs-vocab on clearclip 13.31
  (own 13.43), clearclip-vocab on lposs 15.25 (own 13.67). The searched
  fragility is shared across methods, not per-method search adaptation.
  File: runs/w4f_ans_transfer.json.

## W4h Per-class safety check, 3 new REVA bases (test-300, vs plain baseline)
From existing per_class records (w3b_*_plain vs w3a/w3d REVA sam_reg_vabs):
- ProxyCLIP: person 56.4->48.7 (-7.7), tvmonitor 25.5->11.6 (-13.9),
  pottedplant 9.4->27.0 (+17.6)
- LPOSS: person 56.7->41.3 (-15.4), tvmonitor 14.1->2.1 (-12.0),
  pottedplant 10.1->23.4 (+13.3)
- SC-CLIP: person 56.0->43.0 (-13.0), tvmonitor 22.9->9.9 (-13.0),
  pottedplant 9.4->24.4 (+15.0)
The person/tvmonitor residual-harm signature extends to (and is somewhat
larger on) the 2024-25 family — LPOSS tvmonitor drops to near-zero. Macro
gains remain ~+17..+21. Disclosed in the REVA paper safety discussion; closes
the outstanding "per-class safety for new bases" reviewer item.

## W4i ViT-L/14 REVA six cells (VOC test-300; second-backbone insurance)
- SCLIP:  plain 37.24 / REVA 44.01 / official(pixel) 40.72; VABS-vs-rand +1.8
- NACLIP: plain 36.31 / REVA 51.60 / official(pixel) 50.60; VABS-vs-rand +3.1
REVA gains transfer to ViT-L and exceed the pixel-level official reference on
both methods. Caveats: ViT-L dense baselines are globally weaker (known), no
matched official+SAM bound at ViT-L, VABS negatives were selected with ViT-B
text embeddings (transfer, disclosed).
Files: runs/vitl_{sclip,naclip}_sam.json + existing vitl_* runs.

## W4g External anchor: UNMODIFIED official ProxyCLIP code (mc-lan/ProxyCLIP)
Fresh conda env (torch 2.1/mmcv 2.1/mmseg 1.2.2), their protocol (2048x336
slide, prob_thd 0.2, full VOC val 1449), only the class-name file swapped:
- official name file: mIoU 61.24 (paper reports ~61.0 — sanity anchor OK;
  our style-reimplementation official row 60.03 is within 1.2).
- plain names: 47.26  -> NEG = +14.0 in the authors' own code.
- syn100 names: 41.20 -> synonym drop 6.1 in the authors' own code.
- plain + VABS-64 negatives (name file only): 56.22 -> VABS alone recovers
  64% of the author-code NEG gap (47.26 -> 56.22 vs official 61.24) on
  UNMODIFIED author code. SAM arbitration cannot be added without modifying
  their code, so only the text-side REVA component is anchored (disclosed).
The naming-engineering and synonym-fragility findings replicate on unmodified
author code under the authors' own protocol — closes the reviewers' "no
external anchor" objection for the audit paper. (NEG is smaller than our
protocol's +22.1 partly because their pipeline thresholds background at
prob_thd=0.2, which absorbs part of the plain-vocabulary penalty.)
Files: runs/w4g_proxyclip_{official,plain,syn100}.log.
(A 221-class distractor run on the author code OOMs in their post-processing
at 2048-px inference; patching their code would break the "unmodified"
property, so the distractor external anchor is not claimed.)

Process note: the first w3d launch lost remotely generated ctx60/cocoobj
synonym vocabs to a mid-run rsync --delete (sync discipline violation);
vocabs were regenerated, copied local first, and the queue relaunched.

## W5a Presence-Gated REVA v2 (prereg_w5a_presence2.md) — KILLED (K1)
New signal family (raw-cosine VABS-negative margin + winner-consistency),
explicitly distinct from the killed W4d rank gate. VOC test-300,
ClearCLIP/NACLIP, frozen gate z(s1)>=0 OR s2>=0.3:
- K1 (distractor repair, need all-class >=20): ClearCLIP 3.82, NACLIP 4.23
  (ungated SAM 3.81 / 4.21) -> FAIL on both. KILLED.
- K2 (no harm): PASS, gate is mildly positive on official/plain
  (ClearCLIP official 56.84->57.02, plain 35.75->36.32; NACLIP official
  60.38->60.98, plain 39.30->40.77).
- K3: precision 0.34/0.31 @ recall ~0.98 (need 0.6@0.9) -> FAIL.
Second independent signal family (softmax rank W4d; raw margin+consistency
W5a) that cannot separate vocabulary-item presence at image level. Presence
gating is now dead for the project; distractor collapse remains REVA's open
failure mode. Files: runs/w5a_*.json.

## W5e BA-decomposition protocol (prereg_w5e_ba.md) — GO on decomposition, KILL as standalone paper
Confusion-flow decomposition, 7 methods x {plain, syn100_s0, dis_near200} x
{VOC-21 test-300, COCO-171 val-300} (42 runs, w5e_*.npz).
- Distractor axis: the all-class collapse is >100% metric artifact — GT-present
  mIoU actually RISES on VOC (+4.6..+9.8) while all-class drops 28-37 pts
  (artifact term 34-41 pts); on COCO GT-present is flat (±0.3), artifact 9-15.
  Pixel flows: 64-72% (VOC) / 15-21% (COCO) of GT pixel mass is stolen by the
  injected distractor names (83-88% of that steal comes from the background
  row, per-method range -> quantitative confirmation of the
  background-absorption mechanism; but 29-43% of FOREGROUND pixel mass is
  also stolen); flow INTO the background class is unchanged and inter-class
  confusion DECREASES.
  Post-review wording fixes (R1/R2 incremental, both keep ratings, Accept /
  Weak Accept->Accept-leaning): ">100% artifact" replaced with two separate
  ledgers (pixel-flow vs metric-convention), 87.5% replaced with per-method
  range 83-88%, presence-gating verdict scoped to 2 signal families x 2
  methods x high-recall regime, appendix subset-vs-full-split disclosure
  added.
- Synonym axis: pure inter-class confusion (+0.07..+0.14 flow), zero
  steal/bg, zero artifact — a genuine accuracy loss, unlike the distractor
  axis.
- GO criterion met (artifact+steal >= 50% of drop, dominant term identical
  across all 7 methods and both datasets).
- Standalone-paper criterion KILLED per prereg: BA-corrected (GT-present)
  method ranking is IDENTICAL to uncorrected (Spearman 1.000 on every axis
  and dataset) — the correction changes interpretation, not rankings. The
  decomposition merges into the audit/benchmark appendix as the quantitative
  form of the two-factor finding.
Files: runs/w5e_*.npz, probe_ba_conf.py, probe_ba_decomp.py.

## W6-F3 Vulnerability-prediction law (round-6 F3) — KILLED (prereg_w6f3_law.md)
Frozen text-only signal z(g1 NN-margin) + z(g2 drift-to-plain) vs archived
mIoU rankings (no new segmentation): median per-method Spearman VOC-full
0.03 (kill line 0.5), cocoobj -0.10, ctx60 0.40. The text-geometry signal
that predicted CONFIG quality at fixed vocabulary (E2, rho=0.89) does NOT
rank VOCABULARIES at fixed config. Fourth failure of "text geometry
predicts dense robustness" at the vocabulary level — consistent with the
audit's whitening/geometry-dissociation finding (C7/C8). VocabLint tool
premise dead in this form. Files: runs/w6f3_law.json, probe_law.py.

## W6-F1 RECAL transductive logit debiasing (round-6 F1) — KILLED (prereg_w6f1_recal.md)
Frozen minimal version (mass-flattening prior alpha=0.5, 3 EM iters, bias
on cosine scale /40, bg excluded): hurts EVERYWHERE. VOC test-300:
SCLIP plain 34.8->27.6, syn100 30.6->23.3; NACLIP plain 36.6->30.4,
syn100 29.7->26.8. Both kill clauses hit (plain harm >= 1.0; recovery
negative). Mechanism: VOC class-mass distribution is legitimately far from
flat, so flattening the predicted marginal injects bias instead of removing
name-conditional bias. Note: the three syn100 seeds are byte-identical
vocabularies (100% substitution is deterministic), so the grid is
effectively 2 methods x {plain, syn100}; disclosed. The synonym axis
(pure inter-class confusion) remains un-repaired; a name-conditioned prior
that does not assume mass flatness was NOT tested and would need a fresh
pre-registration. Files: runs/w6f1_*.json, probe_recal.py.

## W6-F2 Cross-family audit transfer (round-6 F2) — GO (prereg_w6f2_crossfam.md)
Grounding DINO base + SAM-B box-prompted masks (Grounded-SAM style harness),
VOC-21 test-300: plain 77.6 (feasibility bar 35 passed by 2x) / syn100 40.0
/ dis_near200 GT-present 73.3, all-class 7.0.
Pattern is sharply DIFFERENT from the CLIP family:
- Synonym axis: -37.6 mIoU (CLIP family: -2..-9) — the grounding family is
  ~5x MORE name-fragile under legal synonyms. Supervised grounding training
  binds masks to exact category surface forms.
- Distractor axis: GT-present only -4.3 (CLIP family: RISES +4.6..+9.8 via
  background absorption); no absorption artifact — grounding has an explicit
  per-query presence decision, but still leaks a few detections to near
  distractors. all-class 7.0 is the same metric-convention effect.
Honest caveats (recorded before any paper claim): VOC-20 categories are
inside GDINO's supervised detection pretraining (COCO/O365/GoldG), so the
plain row is quasi-in-distribution — the synonym drop measures surface-form
binding, not zero-shot failure; single model, single dataset, no ANS axis
yet; style-harness (boxes->SAM masks), not a native segmenter.
GO per prereg (>=5 mIoU pattern difference on synonym axis, both
directions). Files: runs/w6f2_gdino_*.json(+.npz), probe_crossfam.py.

## W6-F2 extension: second model (OWLv2) + second dataset + ANS transfer
GDINO+SAM extension cells: COCO-Object plain 58.4 -> syn100 30.1 (-28.3;
CLIP family drops 2-4 there) — synonym amplification replicates on a second
dataset. ANS(ClearCLIP-searched) on GDINO heldout-200: 39.8 vs plain 78.7
(-38.9), i.e. roughly at its random-syn100 level (40.0): GDINO is uniformly
fragile to ANY renaming.
OWLv2+SAM (same harness): VOC plain 72.5 / syn100 65.3 (-7.1) /
dis GT-present 71.2 (-1.3, all-class 6.8 = convention effect);
COCO-Object plain 53.0 -> syn100 40.1 (-12.9);
ANS heldout-200: 41.6 vs plain 73.4 (-31.8).
Three cross-family conclusions (frozen wording for the paper):
1. The grounding family is NOT monolithic: synonym fragility tracks the
   text-encoder pedigree — BERT-phrase-grounded GDINO collapses (-37.6/-28.3)
   while CLIP-text-tower OWLv2 degrades like the CLIP-surgery family
   (-7.1/-12.9).
2. The searched worst-case (ANS, found on ClearCLIP with WordNet synonyms)
   transfers ACROSS training paradigms: OWLv2 loses 31.8 under ANS vs 7.1
   under random synonyms — random suites underestimate worst-case ~4.5x even
   for a detector never touched by the search.
3. No background-absorption artifact anywhere in the detection family:
   GT-present stays ~flat under 200 injected distractors (explicit per-query
   presence decision); the all-class collapse is purely the metric
   convention, cleanly corroborating Appendix badecomp from a second
   architecture family.
Caveats: box->SAM harness (not native segmentation), VOC/COCO categories
inside both detectors' supervised pretraining, one seed, test-300/h-200.
Files: runs/w6f2_*.json(+.npz), probe_crossfam.py, probe_crossfam_owl.py.

## W6 incremental review (simulated R1/R2) — both keep ratings
R1: Accept maintained. R2: Weak Accept maintained (upgrades on M1/M2).
Fixes applied to the cross-family section per both reviews: lineage claim
downgraded to "consistent with a text-encoder-lineage hypothesis" (n=1 per
lineage, confounds listed) with the ANS-transfer falsifiable prediction made
explicit (GDINO ANS -38.9 ~= its random-syn -37.6, OWLv2 -31.8 >> -7.1);
in-distribution caveat moved into the headline paragraph; within-detector
deltas only; distractor-fire rates added from confusion matrices (17.8/23.6%
of pixels claimed by distractors, ~92% on GT background, fg steal 4.5/8.5%
vs CLIP family 29-43% -> "no FOREGROUND absorption", real open-set FPs
disclosed); ANS rare-synonym/token-length confound disclosed as un-excluded;
box-threshold sweep and subset/seed caveats disclosed; RECAL verdict scoped
to alpha=0.5, "in this setup". Open items for a future run (not blocking):
frequency-matched synonym control (R2-M2), third model per lineage,
threshold sweep.

## W7a Frequency/token-matched control for ANS (prereg_w7a_freqctrl.md) — MIXED, rare-synonym dominated
Controls: for each ANS-changed class, a different WordNet synonym from the
same cosine-[0.70,0.95] pool, matched on CLIP BPE token count (+/-1), 3
seeds (sofa had NO_ALT and keeps the ANS name; disclosed). Held-out-200:
- ClearCLIP: plain 34.4, ANS 13.4 (drop 21.0), controls 19.2/18.5/19.7
  (mean drop 15.3 = 73% of ANS drop).
- OWLv2: plain 73.4, ANS 41.6 (drop 31.8), controls 44.1/49.9/50.3
  (mean drop 25.3 = 80% of ANS drop).
Verdict (per frozen bands: adversarial <=50%, rare-synonym >=80%): OWLv2 at
the rare-synonym boundary, ClearCLIP in the MIXED band. Honest re-framing
applied to the paper: the bulk of the ANS damage AND its cross-paradigm
transfer is RARE-SYNONYM SENSITIVITY (equally-rare token-matched synonyms
already cost 15-25 points, ~2x the random syn100 suite); the search adds a
genuine but modest adversarial increment (~5.7 on ClearCLIP, ~6.5 on
OWLv2). New standalone finding: naming fragility is concentrated in the
rare-synonym tail — random suites underestimate it because they sample the
frequency distribution of WordNet synonyms, not because search finds exotic
adversarial structure. Files: runs/w7a_*.json, probe_freqctrl*.py.

## W7b Text-encoder-lineage go/no-go (prereg_w7b_lineage.md) — NO-GO (hypothesis falsified as stated)
Third models, same box->SAM harness, VOC test-300 plain/syn100:
- OWL-ViT v1 (CLIP text tower): plain 64.0 -> syn 45.0, drop 19.0 —
  OUTSIDE the frozen CLIP-tower band (<=15), nearly in the BERT band
  (>=20). The lineage banding breaks on the very first third model.
- MDETR (RoBERTa, GLIP unavailable offline; substitution recorded in the
  prereg): plain 3.4 = syn 3.4 -> KILL-infeasible clause. Mechanism: MDETR
  is a pure grounding model with no presence decision — it grounds every
  caption somewhere in every image, so per-class captions flood the image
  with boxes. Itself an interface data point (grounding-without-presence
  cannot do vocabulary-conditional segmentation at all) but not usable for
  the banding test.
Conclusion: synonym fragility is NOT cleanly banded by text-encoder
lineage. OWLv2's relative robustness (-7.1) vs OWL-ViT v1 (-19.0) shares
the same CLIP text tower — the difference must come from training recipe
(OWLv2 self-training/scale) not encoder pedigree. The GDINO-vs-OWLv2
contrast stands as a model-level observation; the "lineage law" story is
dead per prereg NO-GO. Paper wording updated (hypothesis paragraph now
reports the falsification). Files: runs/w7b_*.json, probe_crossfam_mdetr.py.

## W7 incremental simulated final review (internal, not a real review)
R1 (D&B): Accept maintained, soundness 8, significance 8. R2 (adversarial):
Accept, soundness 8, significance 7->8, "textbook self-falsification".
Four camera-ready wording items applied same-day: (1) residual adversarial
component written as roughly 5-7 points with control-seed span; (2)
training-recipe attribution softened to "consistent with"; (3) MDETR
exclusion cites the pre-registered infeasibility clause frozen before
synonym results; (4) frequency-stratification lesson promoted to a suite
reuse recommendation. Remaining camera-ready ledger (no new experiments):
anonymous artifact link, LPOSS/SC-CLIP style-reimpl qualifiers (kept),
single-seed/subset caveats (kept), F1 "in this setup" qualifier (kept).

## W7c Cross-lingual axis quick test (prereg_w7c_xlang.md) — MIXED (es structured, zh collapsed)
VOC test-300, fixed dictionary translations frozen before runs.
- SCLIP: zh 3.6 / es 20.7 (plain band ~54.2 -> es retains ~38%).
- OWLv2+SAM: zh 7.8 / es 57.3 (plain 72.5 -> es retains 79%).
Per frozen MIXED clause: Spanish is a structured axis (retention >=50% on
one model AND rank decoupling from the synonym axis — SCLIP is more
synonym-robust than OWLv2 but far less cross-lingually robust, rank flips);
Chinese is a floor collapse for both (<10), boundary note only. Engineering
disclosure: zh queries exceed OWLv2's 16-token CLIP context, truncation was
enabled (itself part of the boundary: English-BPE tokenizers cannot even
represent the query). Promotion of the es axis to the full method matrix is
a W8 candidate, not yet run. Files: runs/w7c_*.json.

## W8 Spanish naming axis, full matrix (prereg_w8_es_axis.md) — PROMOTE
VOC test-300, frozen voc21_es.json (from W7c), 9/9 valid runs:
- Dense CLIP family (es mIoU / plain / retention): MaskCLIP 16.5/31.8/52%,
  SCLIP 20.7/34.8/59%, ClearCLIP 22.0/~37/59%, NACLIP 21.9, ProxyCLIP
  23.4/37.8/62%, LPOSS 23.1/41.2/56%, SC-CLIP 22.3/38.1/59% — tightly
  clustered 52-62% retention regardless of method generation.
- Detectors: OWLv2 57.3/72.5/79% (most robust); GDINO 14.1/77.6/18%
  (worst).
Verdict per frozen criteria: PROMOTE (stable family split, replicating
W7c's decoupling — GDINO is the most synonym-fragile AND most cross-lingual
fragile, OWLv2 the most robust on both, but the dense family's synonym
ranking spread collapses to a uniform ~55-60% band under es: the
cross-lingual axis measures the shared CLIP text tower's multilingual
capability, not method-level design). zh remains a floor collapse boundary
note. Cross-lingual subsection to be added to the audit paper.
Files: runs/w8_es_dense.json, runs/w8_es_gdino.json, probe_w8_es.py.

## Ops note (disclosed): sync --delete wiped remote-only vocab files
sync_ovss.sh rsyncs local->temp-hb with --delete; the freqctrl/es/zh vocab
JSONs originally generated only on temp-hb were deleted by the 10:54 sync
and regenerated (probe_freqctrl.py is deterministic in its seeds; es/zh
were re-copied from the frozen local originals) and are now stored in the
local project so future syncs preserve them. All affected runs had already
completed and their outputs live outside the synced tree.

## W9 (H1) name-level frequency law — NO-GO (killed at go/no-go)
Prereg prereg_w9_h1_freqlaw.md, frozen estimator wordfreq zipf, analysis on
archived confusions only (757 per-name damage observations across 9 models:
7 dense x VOC/COCO syn100 + OWLv2/GDINO syn100 + OWLv2 ANS/freqctrl h200).
Median per-model Spearman(drop, rarity) = -0.22 (GO needed >= +0.5, NO-GO
< 0.3) — the sign is even REVERSED (within the WordNet cosine-filtered
pool, rarer names damage slightly LESS); partial rho controlling token
count -0.15; within-concept median rho 0.0 (9 classes with >=3 names).
Verdict: graded corpus frequency does NOT predict per-name damage. The
W7a result stands but its correct reading is BINARY non-canonical-name
sensitivity (any dictionary synonym off the canonical name is damaging,
matched controls confirm), not a graded frequency law. The audit paper's
"rare-synonym sensitivity" wording makes no graded-law claim, so no paper
change needed; the "law paper" (H1) is dead per prereg. Sixth entry in the
text-side-predictor graveyard (geometry x2, frequency x1). Pivot per
prereg: H2 canonicalization tested as a standalone repair (motivation
weakened, disclosed). Files: stage12_month/w9_freqlaw.json(+.obs).

## W9-H2 name canonicalization repair — NO-GO (discrete word space joins the graveyard)
Prereg prereg_w9_h2_canon.md (frozen rule: WordNet first-sense aliases +
head noun, highest-zipf pick, CLIP-cosine >= 0.80 guard, no tuning).
VOC test-300 dense / heldout-200 OWLv2:
- No-harm check FAILED: plain canonicalized loses 4.1 (SCLIP 34.8->30.7)
  and 3.1 (ClearCLIP 34.9->31.8) — the rule rewrites already-canonical
  names into worse ones (aeroplane->plane ok, but person->someone,
  bus->coach, bicycle->cycle) because the cosine guard cannot distinguish
  canonical from non-canonical inputs.
- Recovery: rare controls ~39-44% (freqctrl_s0 19.6->26.3/25.5), ANS on
  OWLv2 35% (41.6->52.8, +11.3) — real but below the 60% GO line; syn100
  recovery ~0 to negative.
Verdict per frozen bands: NO-GO (mean recovery ~19-26% and plain harmed
> 0.5). The only untested repair family — discrete word-space rewriting —
is now also killed in its frozen training-free form. Honest note: the ANS
+11.3 shows the direction has signal as a DEFENSE against searched/rare
attacks, but a no-harm canonicalizer needs a reliable canonicality signal,
which W9-H1 (frequency) and F3 (geometry) both failed to provide.
Files: runs/w9_canon_dense.json, runs/w9_owl_voc21_ans_clearclip{,_canon}.json.

## W10 (H3) training-recipe natural experiment — MIXED (spread large but non-monotone)
Prereg prereg_w10_h3_recipe.md. MM-Grounding-DINO Swin-T ladder (same
architecture, only training mixture differs), box->SAM harness, VOC
test-300, plain vs syn100_s0 GT-present mIoU:
- T1 O365+GoldG:        plain 67.9, syn 35.7, drop 32.2
- T2 +GRIT:             plain 68.5, syn 46.3, drop 22.2
- T3 +GRIT+V3Det:       plain 63.3, syn 38.5, drop 24.7
All tiers feasible (plain > 35). Spread 10.0 >= 5 but NON-monotone -> MIXED
per frozen bands: reported observationally, no recipe-causation claim.
Observation: holding architecture fixed, the training data mixture moves
synonym fragility by up to 10 mIoU — the largest same-architecture spread
we have seen — with the web-grounding tier (GRIT) the most robust and the
large-vocabulary detection tier (V3Det) partially reversing the gain. This
strengthens the W7b "consistent with a training-recipe effect" wording
(recipe demonstrably matters within one architecture) without licensing a
monotone data-scale story. Substitution disclosed: no O365-only Swin-T was
released; the ladder starts at O365+GoldG. Files: runs/w10_t{1,2,3}_*.json.

## Ops: mmdetection env installed offline on temp-hb
mmengine 0.10.7 / mmcv 2.2.0 (cu121-torch2.4 wheel) / mmdet 3.3.0 with the
mmcv<2.2 version pin relaxed to <2.3 (single sed on mmdet/__init__.py,
standard workaround); fairscale, matplotlib, opencv-headless, bert-base-
uncased HF cache pushed from the sandbox. Checkpoints under
/media/dell/DATA/ovss/checkpoints/mmgdino/.

## W10-H4 Chinese collapse: artifact vs capability — CAPABILITY-DEFICIT
Prereg prereg_w10_h4_zhinterface.md. Adaptation controls on the W7c zh
cells (SCLIP 3.6 / OWLv2 7.8): bare-name queries (no English template)
SCLIP 2.8, OWLv2 8.6; 2-char short forms SCLIP 2.4, OWLv2 7.8. All gains
< 5 (most negative) -> per frozen bands the collapse is genuine
multilingual incapacity of the English-BPE text towers, not a
template/token-overflow interface artifact. Boundary-note wording in the
paper stands, now artifact-controlled. Files: runs/w10h4_*.json.

## W10 reviewer-driven robustness checks (R1/R2 W8-W10 incremental review)
- H1 estimator swap (R2's ②): recomputed the frequency-law Spearman with a
  CLIP-BPE merge-rank rarity proxy (max token merge rank; closer to the
  CLIP training corpus than wordfreq): median rho -0.12, partial (token
  count) -0.14 — sign still reversed, NO-GO robust to the estimator.
- Wording unification (R1's ②): all rare/low-frequency phrasing replaced
  with non-canonical-name sensitivity; suite recommendation rewritten to
  non-canonical stratum or ANS-style search bound.
- xlang section downgraded from "fourth axis" to "cross-lingual probe:
  Spanish case study"; lineage-vs-multilingual disambiguation added;
  MM-GDINO wording tightened to "not monotone in this three-point sweep".
- Spanish best-of-3 translation control frozen
  (prereg_w10_es_bestofk.md) and running (R2's ①, major).

## W10 Spanish best-of-3 translation control — TRANSLATION-CHOICE (per frozen band)
Prereg prereg_w10_es_bestofk.md. SCLIP (dense representative) per-vocab
GT-present mIoU: es 20.66 / alt1 19.10 / alt2 14.65; per-class best-of-3
oracle 30.03 = 86.4% of plain English (34.75) -> crosses the frozen 85%
TRANSLATION-CHOICE line. OWLv2: 57.3 / 38.9 / 34.8, best-of-3 61.96 =
84.4% of plain 73.4 (just under the line, same direction). Verdict: the
52-62% single-translation dense band substantially reflects sensitivity
to WHICH legal translation is chosen — the Spanish probe is largely the
naming-choice fragility phenomenon in another language, not (for Spanish)
a multilingual capability deficit. The family split (detector contrast)
is preserved across translations. Chinese remains a genuine capability
collapse (W10-H4 artifact controls). Paper xlang section rewritten per
the frozen band. Files: runs/w10_es_bestofk_sclip.json,
runs/w10_owl_es_alt{1,2}.json(.npz).

## W11-J2 LLM canonicality judge — NO-GO
Prereg prereg_w11_j2_llmjudge.md (frozen prompt + judgments in
perturbed_vocabs/w11_llm_judgments.json, produced before any run).
Concept-blind LLM canonicality gate + rewrite, no CLIP guard.
Dense (VOC test-300): plain no-harm PASSES (SCLIP 34.75->34.38,
ClearCLIP 34.89->34.48, both within -0.5) — the gate does leave canonical
vocabularies almost alone. But recovery is NEGATIVE on every damage arm:
syn100 SCLIP 30.55->26.89 (-3.7), ClearCLIP 29.03->24.50 (-4.5);
freqctrl SCLIP 19.56->18.98, ClearCLIP 19.58->18.65; OWLv2 ANS heldout
41.6->38.35 (-3.3). Mechanism: without the intended visual concept the
judge resolves names to their dominant word sense (kat->khat the plant,
bounder->scoundrel, fowl->bird...) and rewrites legitimate synonyms across
senses. Seventh graveyard member: word-space repair now dead under three
signal families (geometry, corpus frequency, LLM world knowledge) — the
missing ingredient is the intended concept, which is exactly what a
vocabulary-only repair cannot know. Runs: runs/w11_j2_dense.json,
runs/w11_j2_owl_ans_llmj.json.

## W11-J1 alias-diversity hypothesis — NO-GO
Prereg prereg_w11_j1_alias.md. Corpora: GoldG (mdetr final_mixed +
flickr mergedGT train captions), GRIT 1M-caption sample (coyo_0 shard,
frozen), V3Det 13204 names, O365 365 names. Alias pool = frozen WordNet
cosine pool + canonical name.
Recipe-level mean alias entropy: T1(GoldG) 0.294, T2(+GRIT) 0.422,
T3(+V3Det) 0.422 — GRIT does add alias diversity (+43% entropy) in the
direction of the T1->T2 robustness gain, but V3Det adds none, so the
entropy index cannot explain the T2->T3 reversal (GO needed T2>=T3>T1
with class-level rho >=0.4).
Class-level: Spearman(per-class T1->T2 damage reduction, GRIT alias
entropy) = 0.107 (p=0.65, n=20) < 0.25 -> NO-GO.
Honest reading: alias diversity as a graded per-class predictor of
recipe-driven robustness fails, same signature as the dead H1 frequency
law (corpus-level contrast exists, per-name/per-class grading does not).
The observational corpus contrast (web grounding text is alias-richer
than detection labels) may be kept as a descriptive note only.
Files: stage12_month/w11_j1_alias.json, runs/w11_alias_pool.json.

## W11-J3 translation-choice sensitivity beyond Spanish — MIXED
Prereg prereg_w11_j3_translation.md (de/ru, k=3 frozen dictionary
translations, SCLIP + OWLv2+SAM, VOC test-300, GT-present).
Retention vs plain English (SCLIP 34.75 / OWLv2 72.5):
- German: SCLIP default 63.7% -> best-of-3 77.7% (gap 14.0 pts);
  OWLv2 81.6% -> 94.3% (gap 12.7).
- Russian: SCLIP 19.0% -> 29.9% (gap 10.9); OWLv2 5.9% -> 8.5%
  (gap 2.6) — Cyrillic floor collapse on OWLv2, same signature as
  Chinese (H4): non-Latin script + English BPE.
- Spanish (W10): SCLIP 59.5% -> 86.4% (gap 26.9); OWLv2 79% -> 84.4%.
Frozen bands: only Spanish crosses the 15-pt GO line (1 of 3 languages),
but both new languages exceed the 8-pt NO-GO floor on the dense model ->
MIXED. Honest reading: translation-choice sensitivity is real and
pervasive (10-27 pt oracle-default gaps wherever the script is
representable) but its magnitude is language-dependent; and Russian adds
a second script-floor data point. Reported observationally as an
extension of the Spanish case study, no standalone protocol-paper claim.
Files: runs/w11_j3_sclip_{de,ru}.json, runs/w11_j3_owl_*.json(.npz).

## W11-J4 cross-model box-evidence shield — NO-GO on the distractor axis
Prereg prereg_w11_j4_boxshield.md. SCLIP + OWLv2 box support (thresh 0.2,
archived crossfam value), VOC test-300, dis_near200 (21+200):
dense all-class 4.23 -> shielded 4.27 (GO needed >=25, NO-GO <15).
Mechanism: OWLv2 itself fires on near-distractor names (W7: 23.6% of
pixels claimed), so the box-support set retains enough distractor entries
to keep the all-class denominator polluted; the detector's robustness is
in WHERE it fires (background, not foreground), which a presence-level
shield cannot exploit. Fourth signal family dead on this axis
(softmax-rank, VABS-margin/top-K, LLM judge n/a, external box evidence).
Distractor collapse remains REVA's open problem.
OUT-OF-PREREG OBSERVATION (not a claim, candidate for next wave): the
same shield applied to the PLAIN 21-class vocabulary lifts SCLIP
34.75 -> 40.80 (+6.0) — image-conditioned vocabulary subsetting by an
external detector substantially helps dense CLIP on clean vocabularies.
Needs its own prereg (cross-dataset, cross-method, trivial-baseline and
harm controls) before any use. Files: runs/w11_j4_shield_dis200.json,
runs/w11_j4_shield_plain.json.

## W11-J5 detector-guided vocabulary pruning — GO
Prereg prereg_w11_j5_vocabprune.md (frozen immediately after the J4
out-of-prereg observation; J4 plain cell not reused as evidence).
OWLv2 box-support pruning (thresh 0.2, untuned) of the dense argmax,
9 cells {SCLIP, ClearCLIP, NACLIP} x {VOC, ctx60, cocoobj} test-300:
gains +6.1/+2.0/+2.6, +4.7/+1.4/+3.9, +9.2/+3.2/+6.4 — mean +4.4 >= +2,
no cell harmed (GO also needed synonym harm <2: syn100 SCLIP/VOC
30.55 -> 35.96, i.e. pruning HELPS the synonym arm too; pruned synonym
drop 4.8 vs unpruned 4.2, essentially unchanged). Disclosures: adds a
second model at inference; OWLv2+SAM standalone remains higher on VOC
(72.5) — the claim is dense-vs-dense-pruned; ctx60 stuff classes still
gain despite boxes being a poor fit. Composability with VABS+SAM
(REVA) not yet run — required before paper integration.
Files: runs/w11_j5_prune.json (copy in stage12_month/).

## W11-J5 composability cell — GO (stacks with REVA)
SCLIP/VOC test-300: REVA (VABS64+SAM) 57.04 -> +OWLv2 pruning 59.40
(+2.4). Pruning composes additively with VABS+SAM; combined pipeline
reaches 59.4 vs official-names pixel reference 58.8. Third component
candidate confirmed on this cell; cross-method/dataset composition table
still pending before full REVA-paper table integration.
File: runs/w11_j5_reva.json.

## W11 reviewer remediation — GDINO Spanish best-of-3 oracle
R2's remaining must-fix: does translation choice explain the GDINO 18%
vs OWLv2 79% family split? GDINO es_alt1 9.37, es_alt2 5.54 (default
14.11); per-class best-of-3 oracle 19.73 = 25.4% retention vs plain 77.6.
Oracle recovery raises GDINO only 18%->25% (vs SCLIP 59->86%, OWLv2
79->84%) -> the family split is NOT a translation-choice artifact; the
"under default translations" qualifier can be strengthened with this
bound. Files: runs/w11_gdino_es_alt{1,2}.json(.npz).

## W11-J5 composability — two further cells (VOC test-300)
ClearCLIP REVA 54.50 -> +prune 57.03 (+2.5); NACLIP REVA 57.53 -> +prune
61.11 (+3.6). All three REVA+prune cells positive (SCLIP +2.4); pruning
gain persists on top of VABS+SAM across methods.
Files: runs/w11_j5_reva_{clearclip,naclip}.json.

## W12-K1a absent-class leakage mechanism — NO-GO at class level (corpus-level supported)
prereg_w12_k1_mech.md. SCLIP/VOC test-300, plain->syn100_s0 leaked
GT-foreground pixels: 55.4% go to ABSENT classes, 41.6% to background,
only 3.0% to present classes -> corpus-level mechanism supported (>=50%
line met). But class-level correlation between absent-leak fraction and
J5 pruning gain: Spearman -0.20 (p=0.48, n=14) < 0.25 -> frozen NO-GO
(same signature as J1/H1: pooled contrast real, class-level graded
predictor dead). Consequence per frozen plan: K1 DGVP does not stand as
an independent mechanism paper; J5 stays a REVA-paper component section;
the 55%-absent accounting can be reported descriptively.
File: runs/w12_k1_mech.json.

## W12-K3 GT-presence oracle on distractor axis — NO-GO, and the criterion itself is the finding
prereg_w12_k3_oracle.md. dis_near200, three models. Oracle pruning to
GT-present classes: all-class stays ~4 (SCLIP 4.23->4.14, ClearCLIP
3.88->3.99, NACLIP 4.04->4.66) — far below the 40% recovery line ->
frozen NO-GO. Honest reading: the frozen all-class criterion is
structurally unattainable — with a fixed 221-class denominator, 200
never-predicted classes contribute IoU 0, capping all-class mIoU at
~(21/221)*ceiling ~= 4.3 for ANY method INCLUDING a perfect presence
oracle. GT-present effect of the oracle is small and mixed (SCLIP 44.5->
43.5, ClearCLIP 40.9->42.0, NACLIP 42.5->49.0). Conclusion: the
distractor "collapse" is not repairable by any vocabulary filtering —
not because presence is unknowable (four dead signal families) but
because the all-class convention itself charges the denominator. The
restatement is about the METRIC, not presence estimation; this closes
the distractor-repair line and strengthens the BA-decomposition chapter.
Prereg criterion mis-specification disclosed as such (frozen before run,
verdict kept).
Files: runs/w12_k3_oracle_{sclip,clearclip,naclip}.json.

## W12-K4 red team + ADE-150 scale — MIXED (attack) / GO (scale, direction reversed)
prereg_w12_k4_redteam.md. (a) ANS attack under pruning (heldout-200):
SCLIP ANS 14.29->21.35 with pruning, plain 34.67->40.38; damage 20.4->
19.0, recovery 34.7%. ClearCLIP: 13.43->20.70 / 34.45->39.14; recovery
34.6%. Both in (30,60) -> frozen MIXED: pruning lifts the attacked arm
+7 but is NOT a defense — the searched vocabulary damages CLIP scoring
of present classes, which pruning cannot restore (consistent with J4).
(b) ADE-150 scale cells: SCLIP 14.10->16.76 (+2.66), NACLIP 15.39->
17.87 (+2.48); mean +2.57 >= +2, none harmed -> GO. But gain does NOT
grow with vocabulary size (VOC +6..+9 vs ADE +2.5): the corpus-level
absent-mass prediction fails at scale — plausibly detector recall
degrades on 150 finer-grained queries. Both facts to be disclosed in
the REVA J5 subsection. File: runs/w12_k4.json.

## W12-K2 contamination-curve protocol — NO-GO (frozen criteria), with a sharpening observation
prereg_w12_k2_contamination.md. VOC test-300, SCLIP+NACLIP, near/random
pools injected at n=0..150. GT-present mIoU does not DROP at all — it
RISES monotonically-ish with contamination (SCLIP near 34.8->44.0,
rand 34.8->46.6; NACLIP near 36.5->41.6, rand 36.5->45.3): injected
absent names absorb background pixels and raise GT-present precision,
the same background-absorption mechanism as the distractor axis, now
shown to be graded and pool-dependent (random absorbs MORE than near).
Frozen GO required a >=3 GT-present DROP -> NO-GO; the effective-
vocabulary confound reduces entirely to the convention/absorption story
already covered by the BA decomposition, and a separate protocol paper
is not warranted. Sharpening for the audit paper: even 10 random absent
names buy +6..+9 GT-present mIoU — "expansion hurts/helps" is set by
convention + background absorption from the very first injected name.
Files: runs/w12_k2_{sclip,naclip}.json.

## W13-L1 metric-convention go/no-go — NO-GO (protocol paper not warranted)
prereg_w13_l1_metric.md. Offline recompute from archived w5e confusion
records: 7 methods x {plain, syn, dis_near200} x VOC under three
conventions (fixed-denominator / NaN-exclusion / GT-present). Absolute
values move enormously on the distractor axis (fixed ~4 vs present
37-46; NaN ~4 too, because raw predictions DO fire on distractors so
their unions are nonzero — NaN exclusion only rescues the collapse
after pruning removes those predictions). But method-ranking Spearman
is 1.000 for every convention pair on every axis -> frozen NO-GO (all
pairs > 0.95, no new conclusion flips; the presence-gating mis-kill
remains the single pre-existing flip). Conventions change absolute
values and axis interpretation, not method comparisons — consistent
with the BA-decomposition invariance. L1 is demoted per prereg to an
audit-paper note; no protocol paper.

## W13-L2 J5 defensive backfill — top-k NON-TRIVIAL / soft pruning MIXED
prereg_w13_l2_softprune.md. (a) Matched-budget CLIP image-level top-k
baseline: VOC reaches ~80% of the pruning gain (SCLIP +4.8 vs +6.1,
NACLIP +7.4 vs +9.2) but is NET NEGATIVE on ADE-150 (SCLIP -3.9 vs
+2.7, NACLIP -3.9 vs +2.5). Four-cell average far below the 50% line ->
NON-TRIVIAL: the detector component is what makes pruning safe at
scale; CLIP's own image-level ranking cannot identify present classes
in a 150-class vocabulary. (b) Soft pruning: lambda=0.3 best; improves
ADE over hard by +0.6/+0.6 (below the >=1 SOFT-WINS line) while losing
0.1 (SCLIP) / 1.7 (NACLIP) on VOC -> frozen MIXED: mild large-vocabulary
mitigation, not free. Both disclosed in the REVA J5 subsection.
File: runs/w13_l2.json.

## W13-L3 filler free lunch — NO-GO (background calibration in disguise)
prereg_w13_l3_filler.md. SCLIP/VOC test-300, n=50 fillers. Random real
nouns +11.8 GT-present; pseudo-words +6.2 (52% of R — between the 70%
GO and 40% NO-GO lines); filler-captured pixels 91.7%/91.8%
GT-background. But the trivial control kills it: a single scalar
background-logit boost of +0.02 cosine units reaches 47.5 (> the R arm
46.5, i.e. >100% of the R gain, vs the frozen 85% NO-GO line) -> NO-GO:
the free lunch is dominated by background under-calibration that a one-
parameter boost fixes better than 50 injected names. (VABS arm skipped:
VABS negatives live as background aliases, i.e. the same mechanism
family as the boost control.) Interpretation kept honest: this both
kills the L3 law paper AND sharpens the audit story — the anomalous
expansion gain is background calibration, not semantics; sweeping a
background boost is the correct baseline any absorption claim must
beat. File: runs/w13_l3.json.

## W13-L5 VABS vs scalar background boost — VABS-DISTINCT, but boost is a strong disclosed baseline
prereg_w13_l5_vabsboost.md (self-attack triggered by L3). VOC, dev=first
100 imgs (boost selection), eval=next 200, matched fold convention.
SCLIP: plain 34.7 / plain+boost(b*=0.03) 50.7 / VABS 53.5 /
VABS+boost(0.01) 54.6. NACLIP: 36.6 / 48.4(b*=0.05) / 52.8 / 53.0.
Frozen verdict: VABS-DISTINCT (boost <= VABS-2.0 on both models; boost
adds only +0.2..+1.1 on top of VABS). Honest disclosure required: the
dev-tuned scalar boost reproduces 73-85% (recomputed: SCLIP 85.1%, NACLIP 72.8%) of the VABS text-side gain on
VOC — but it needs a labeled dev split to pick b*, while VABS is
label-free and untuned; and b* is brittle (0.08 collapses SCLIP to 33).
To be added to the REVA paper as a baseline row + framing sentence.
Also note: L3's earlier "boost > fillers" compared against SEPARATE-
ENTRY fillers, not fold-convention VABS; under matched folding VABS
clearly wins. Files: runs/w13_l5_{sclip,naclip}.json.

## W13-L4 detector granularity — MIXED (granularity account refuted in sign)
prereg_w13_l4_granularity.md. ADE-150 test-300, OWLv2 thr 0.2, 86 valid
classes (>=5 images), all with WordNet synsets. rho(depth, presence
recall) = +0.166 — WRONG SIGN for the hypothesis (deeper/finer labels
have slightly higher recall), partial rho controlling log pixel
frequency +0.135; rho(recall, per-class J5 gain) = +0.05. GO required
rho <= -0.3 AND rho(recall,gain) >= 0.4: decisively failed; strict
NO-GO clauses also not met -> MIXED by the letter, but the
granularity-degradation explanation of the ADE gain shrinkage is
unsupported. Mean presence recall 0.67: a third of GT-present classes
get wrongly pruned yet pruning still nets +2.7 — the shrinkage
mechanism remains open; paper wording to be softened ("plausibly
because detector recall degrades on finer-grained queries" removed).
File: runs/w13_l4.json.

## W13-L5t boost cross-dataset transfer (prereg_w13_l5_vabsboost.md#addendum) — NOT-TRANSFER
VOC-dev-selected boost (SCLIP b*=0.03, NACLIP b*=0.05) applied unchanged, test-300, GT-present mIoU:
- sclip/ade150: plain 14.10 (no bg class -> boost inapplicable on plain), vabs 13.13 -> vabs+boost 11.83 (harm -1.30)
- sclip/ctx60:  plain 26.63 -> boost 26.31 (harm -0.32); vabs 27.25 -> vabs+boost 21.47 (harm -5.78)
- naclip/ade150: plain 15.39, vabs 14.74 -> 13.32 (harm -1.42)
- naclip/ctx60: plain 27.46 -> boost 26.58 (harm -0.88); vabs 27.98 -> 18.35 (harm -9.63)
Frozen verdict: NOT-TRANSFER — the labelled-dev-tuned scalar harms on every transfer cell;
also structurally inapplicable on ADE plain (no background class to boost).
Reading: the boost is a dataset-specific calibration constant; VABS's advantage includes
transfer-without-retuning (VABS itself: +0.5/+0.6 on ctx60, -0.7/-1.0 on ADE test-300,
consistent with the disclosed ADE null boundary). Single seed, test-300.

## W13-M1 pruning profit/loss ledger (prereg_w13_m1_ledger.md) — STRUCTURE FOUND
Pixel ledger (test-300, THRESH=0.2): A = dense-correct pixels destroyed by pruning their GT class,
B = dense-wrong -> pruned-correct corrections.
- sclip/voc21:  A 1.01M, B 7.86M (B:A 7.8); mispruned present classes 69/502 = 13.7%
- naclip/voc21: A 1.12M, B 8.09M (B:A 7.2); 13.7%
- sclip/ade150: A 2.64M, B 7.35M (B:A 2.8); mispruned 850/2483 = 34.2%
- naclip/ade150: A 3.93M, B 7.40M (B:A 1.9); 34.2%
Verdict (frozen >=2x rule met): the VOC->ADE shrinkage is a THICKER LOSS SIDE, not a thinner
profit side — B is nearly constant, A grows 2.6-3.5x as the mispruned present-class share rises
13.7% -> 34.2%. Fault-tolerance structure confirmed: mispruned present classes are the ones dense
CLIP already segments poorly (dense IoU 27.8/29.6 vs 47.1/50.3 kept on VOC; 10.2/14.1 vs 28.3/30.4
on ADE) and are smaller — deleting them costs less than their count suggests, which is why 1/3
mispruning still leaves a net gain. Descriptive accounting; single seed, test-300, 2 models.

## W14-N3 fixed hand list vs VABS off-VOC (prereg_w14_n3_handlist.md, test-300, single seed) — VABS-ADAPTIVE
Trigger: rebuttal drill Attack 1 (on VOC the 26-entry hand list matches/beats REVA under SAM pooling;
adaptivity untested where the list's VOC assumptions break). Pixel arms, SCLIP+NACLIP.
Run: /media/dell/DATA/ovss/runs/w14_n3.json, log w14n3.log.
GT-present mIoU:
  sclip/ade150:  plain 14.10  hand 13.93  vabs 13.13
  naclip/ade150: plain 15.39  hand 15.36  vabs 14.74
  sclip/cocoobj:  plain 22.57  hand 30.48  vabs 31.81
  naclip/cocoobj: plain 22.08  hand 30.65  vabs 32.81
Frozen verdict: VABS-ADAPTIVE — on COCO-Object VABS beats the verbatim hand list by +1.3/+2.2
(> +0.5 line on one dataset); HANDLIST-DOMINATES fails (hand < vabs-0.3 on COCO).
Honest notes: on ADE-150 both arms are null-to-slightly-harmful (no background channel; the hand
list is ~= plain, VABS slightly below) — the disclosed ADE null extends to the hand list. On COCO
both arms share partial stuff->background leak (hand26 contains sky/tree/grass etc.); the
comparison is leak-matched in kind, not exactly in degree. Plain baselines match archived runs
(22.6/22.1 and 14.1/15.4). Single seed, test-300.

## W14-N4 PAMR post-processing effect-size check (prereg_w14_n4_pamr_effect.md, VOC test-300, single seed) — SURVIVES
Trigger: rebuttal drill A1 (effect sizes measured without publication-grade post-processing).
SCLIP/NACLIP, PAMR-10 refinement on pixel probs, GT-present mIoU:
  sclip:  official 55.42->58.56, plain 34.75->35.86, syn100 30.55->31.99
  naclip: official 55.02->61.02, plain 36.54->38.87, syn100 29.67->31.99
Effect sizes (avg of 2 models): NEG no-PAMR 19.6 -> PAMR 22.4 (+15%, i.e. post-processing
AMPLIFIES the naming gap, helping engineered vocab more than plain); syn-drop 5.5 -> 5.4 (-3%).
Both within frozen +/-30% band -> SURVIVES. Runs: /media/dell/DATA/ovss/runs/w14n4_*.json.
Bounds the post-processing axis only (no multi-scale / official repo stack), disclosed as such.
