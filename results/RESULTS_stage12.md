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

## W14-N5 REVA under perturbed vocabularies + 3-seed random-negative control (prereg_w14_n5_reva_axes.md, VOC test-300, SCLIP+NACLIP) — Part A: SYN-COVERED / ANS stress bound; Part B: seed-sensitive advantage
Trigger: blind REVA R1 (title over-claims "vocabulary-robust" with only the plain axis tested;
random-negative control single seed). VABS negatives regenerated conditioned on each perturbed
vocabulary (vabs.py, scene lexicon, M=64, tau=0.90, frozen recipe — selected sets differ from the
plain-conditioned set). Arms: pix_vabs / sam_reg_vabs (REVA) / sam_reg_rand (here = SAM-region on
the bare perturbed vocabulary, the matched baseline). Runs: /media/dell/DATA/ovss/runs/w14n5_*.json.
Part A (all-class mIoU, test-300):
  syn100: sclip  bare-SAM 32.07 -> REVA 49.91 (+17.8); naclip bare-SAM 31.63 -> REVA 46.78 (+15.2)
    Plain-regime reference gains (bare-SAM plain -> REVA): sclip 36.60->57.04 (+20.4),
    naclip 39.59->57.53 (+17.9). REVA retains ~85-88% of its plain-regime gain under syn100 ->
    per frozen expectation the synonym axis can be claimed as covered for the *gain*; the
    residual ~7-mIoU synonym cost itself is NOT repaired (49.9 vs 57.0), disclosed as such.
  ANS (clearclip-searched vocab): sclip bare-SAM 15.27 -> REVA 25.09 (+9.8);
    naclip 14.66 -> REVA 23.85 (+9.2). REVA lifts but does not rescue adversarial vocabularies
    (end level ~25 vs plain ~57) -> reported as a stress bound, not a robustness claim.
Part B random-negative seeds (SAM-region selection advantage vabs - rand, VOC test-300):
  sclip:  s0 +3.76, s1 -0.05, s2 +1.40 -> mean +1.70 (range -0.05..+3.76)
  naclip: s0 +3.56, s1 +1.03, s2 +2.83 -> mean +2.47 (range +1.03..+3.56)
  Positive in 5/6 cells but the 3-seed mean is smaller than the seed-0 headline and the spread
  is comparable to the 3.1 subset noise floor -> paper claim softened from "clearly beats matched
  random negatives" to a seeded mean +/- range with a noise-floor caveat. Single test subset;
  seeds vary the random negative set only.
Verdict per frozen rules: title/abstract stay narrowed to plain-vocabulary robustness (done in
this revision); syn100 coverage reported as a positive secondary result; ANS as stress bound.

## W15 LPOSS author-code anchor (prereg_w15_lposs_anchor.md, full VOC val 1449) — DIRECTION REPLICATES / NEG BELOW LINE
Unmodified official LPOSS release (CVPR'25), lposs.yaml, own protocol (2048x448 slide,
DINO+MaskCLIP ViT-B-16 laion2b), only the CLASSES list changed via external wrapper
(w15_lposs_anchor.py). Logs: /media/dell/DATA/ovss/runs/w15_lposs_{official,plain,syn100}.log.
  official (authors' 26-term bg expansion): 61.14  (published 61.2; our style-reimpl official row 60.5)
  plain names:                              55.14  -> NEG +6.0
  syn100 (s0, same suite as audit):         47.70  -> synonym drop 7.4 (vs 4.3 on our style-reimpl)
Frozen verdict: direction replicates on both axes (naming engineering positive, synonym
fragility strong — larger than our-protocol estimate), but author-code NEG (+6.0) is below
the frozen +8 line and far below the style-reimpl-protocol NEG (19.6): LPOSS's label-propagation
smoothing + own pipeline absorbs much of the plain-vocabulary penalty. Per prereg, the audit
paper's LPOSS leaderboard statements are re-anchored with these numbers and effect-size
modulation is disclosed (as with ProxyCLIP: author-code +14.0 vs our +22.1). ANS/worst-case
statements about the LPOSS-style row remain our-protocol-only and stay flagged.
Env note: dedicated conda env (torch 1.12.1+cu113, mmcv-full 1.6.0, faiss-gpu 1.8.0);
laion2b CLIP weights transferred to the machine's HF cache (HF unreachable from it).

## W15-B full-split 3-seed random-negative control (prereg_w15_fullsplit_randneg.md) — GO: consistent positive
VOC-21 full dev-excluded split (1349), probe_d1sam, frozen VABS recipe; runs
w15_fullrand_s{0,1,2}_{sclip,naclip}.json (artifact_release/runs/).
Region-arbitration selection advantage (sam_reg_vabs - sam_reg_rand):
  SCLIP : vabs 58.71; rand s0 53.63 / s1 57.47 / s2 55.77 -> +5.08 / +1.24 / +2.94 (mean +3.09)
  NACLIP: vabs 59.18; rand s0 54.45 / s1 57.14 / s2 55.16 -> +4.73 / +2.05 / +4.02 (mean +3.60)
6/6 cells positive, means >= +1.0 -> frozen criterion met: claim upgraded to
"consistent positive selection advantage on the full split" (range +1.2..+5.1);
test-300 3-seed spread disclosure retained as the subset-level caveat.

## W15-C distractor-set multi-seed control (prereg_w15c_distractor_seeds.md, VOC test-300) — MIXED per frozen lines
3 draws of 200 near distractors (s0 = shipped deterministic first-200 of sorted
near stratum; s1/s2 = seeded random samples from the same frozen stratum).
Runs: w15c_dis_s{1,2}_{sclip,naclip}.json + archived s0 (w3b_*_dis_near200).
  all-class (fixed denominator): SCLIP 4.23/4.55/4.40 (spread 0.32); NACLIP 4.04/4.44/4.31 (0.40)
  GT-present:                    SCLIP 44.53/47.87/46.28 (spread 3.34); NACLIP 42.52/46.69/45.37 (4.17)
Frozen verdict: all-class collapse is stable across draws (well within the 2-mIoU
line) -> "stable across distractor draws" applies to the collapse; GT-present
spread slightly exceeds the 3-mIoU line (3.3-4.2) -> GT-present magnitude reported
as draw-dependent within ~4 mIoU (the deterministic first-200 draw is the harshest).
Convention-gap conclusion (10x gap all-class vs GT-present) unaffected: holds in
every draw.

## W15-D negative-claim mechanism accounting (prereg_w15d_negclaim_mech.md, VOC test-300, region arbitration) — HYPOTHESIS SUPPORTED
Runs: w15d_mech_{sclip,naclip}.json (frozen recipe, rand = s0 matched negatives).
Background-claim quality vs GT background:
  SCLIP : recall vabs 89.2 vs rand 84.2 (+5.0); precision 92.2 vs 90.9 (+1.3)
  NACLIP: recall vabs 91.2 vs rand 86.5 (+4.7); precision 91.3 vs 90.5 (+0.8)
Winning-row share (pixel level, pre-arbitration): VABS negatives win 35.6%/32.6%
of pixels vs random negatives 17.1%/19.0% — vocabulary-conditioned negatives are
competitive against foreground classes where random negatives are not.
Frozen criterion met in both variants (recall +>=3 with precision within 2/above):
the selection advantage is background RECALL gained at no precision cost — random
negatives under-claim background, leaving it to leak into objects. Descriptive,
single rand seed, test-300.

## W15-E per-class harm predictor (prereg_w15e_perclass_predictor.md, offline) — NULL (negative result preserved)
Predictor = max CLIP text cosine of each foreground class to the 64 VABS negatives;
outcome = per-class REVA delta (sam_reg_vabs full-split s0 minus plain pixel baseline
w4a_voc21full_*). Spearman over 20 classes: SCLIP rho=+0.37 (p=0.11), NACLIP
rho=+0.39 (p=0.09) — wrong sign vs the frozen hypothesis (closer negatives were
hypothesized to HURT) and |rho|<0.40 in both variants -> NULL per prereg; no
post-hoc predictor tried. Run: w15e_predictor.json. Practical note: text-side
proximity to the negative set does not identify at-risk classes; the per-class
safety table remains the deployment check.

## W16 ANS search-order multi-seed control: STABLE (prereg_w16_ans_orderseeds.md)
SCLIP, VOC, greedy ANS with alphabetical vs 2 seeded coordinate orders
(runs: w16_ans_order_{alpha,s0,s1}_sclip.json):
- search-100 mIoU: 14.42 / 13.69 / 14.14; heldout-200: 14.51 / 13.15 / 13.38.
- spread 1.36 < frozen 3.0 criterion -> bound is search-order robust.
- vocab overlap with alphabetical: 16/21 (s0), 16/21 (s1) -- different
  vocabularies, same damage level (vs heldout plain 34.67).
- Prereg amendment (disclosed): assumed archived alphabetical SCLIP run did not
  exist (original W4f searches were ClearCLIP/LPOSS); alphabetical SCLIP search
  run fresh under identical protocol.
Paper: order-robustness sentence added to ANS paragraph.

## W16-B SC-CLIP author-code anchor: DIRECTION + MAGNITUDE REPLICATE (prereg_w16b_scclip_anchor.md)
Unmodified official SC-CLIP (TIP'25) release, own mmseg protocol, full VOC val
1449 (logs: w16b_scclip_{official,plain,syn100}.log):
- authors' name file: 64.60 (published 64.6 -- exact reproduction);
- plain names: 42.97 -> NEG +21.6;
- syn100: 37.99 -> synonym drop 5.0.
Verdict per frozen criteria: direction replicates on both axes; unlike
ProxyCLIP (+14.0 vs +22.1) and LPOSS (+6.0 vs +19.6), SC-CLIP's author-code
NEG matches the local-protocol +20.1 -- its official name file is the most
heavily engineered in our set (26-item background list + multi-alias entries)
and its pipeline has no absorbing mechanism. "SC-CLIP unanchored" caveat
removed from paper; all three reimplemented leaderboard rows now anchored.

## W16-C VABS on author code + random control: BOTH CRITERIA PASS (prereg_w16c_vabs_authorcode.md)
Unmodified official SC-CLIP, full VOC val 1449, only name file changed
(logs: w16c_scclip_{vabs,rand}.log; plain baseline from W16-B):
- plain 42.97 -> plain+VABS64 59.62 (+16.7, >= frozen +3 bar; 77% of the
  author-code NEG gap to 64.6 recovered);
- plain+matched random-64 56.53 -> VABS selection advantage +3.1 (> 0 bar),
  numerically matching the W15-B full-split means (+3.1/+3.6) in our stack.
Honest note: random negatives alone recover +13.6 -- most of the pixel-level
gain is negative-expansion per se; the selection advantage is the same +3
seen in-stack. Scope: anchors the VABS component only, not SAM arbitration.
Paper: added to REVA external-anchor paragraph.

## W16-D SC-CLIP author-code anchor, 2nd dataset (COCO-Object): AUDIT AXES REPLICATE; VABS TRANSFERS; SELECTION ADVANTAGE NULL (prereg_w16d_scclip_cocoobj.md)
Unmodified official SC-CLIP, cfg_coco_object, full val 5000 (masks converted
with the repo's own mapping, val only; logs w16d_scclip_*.log):
- official 37.72 (published 37.7 -- exact reproduction);
- plain 34.40 -> NEG +3.3 (>0, direction replicates; first official-vs-plain
  measurement for COCO-Object anywhere in the project -- our protocol has no
  official COCO-Object vocab);
- syn100 27.89 -> synonym drop 6.5 (replicates, larger than VOC's 5.0);
- plain+VABS64 37.94 (+3.5 >= +3 bar, reaches official parity);
- plain+rand64 37.96 -> selection advantage -0.02: FAILS the frozen >0
  criterion. Honest verdict: on COCO-Object author code, any negative
  expansion recovers the (small, +3.3) gap fully; vocabulary-conditioned
  selection adds nothing there. Consistent with gap-vs-recovery: the
  selection advantage is measurable where background headroom is large (VOC)
  and vanishes where NEG is small.
Engineering disclosure: the repo's alias-max postprocess OOMs at 145 query
rows on 24GB; replaced with a mathematically identical chunked max
(regression-verified: VOC official 64.60 unchanged, w16d_regress.log).

## W16 incremental review (both reviewers) + wording fixes applied
R1 (audit): numbers verified; required qualifiers applied: (a) "method-dependent
modulation" scoped to the three anchored methods; (b) NEG dataset-dependence
warning sentence added (VOC +21.6 vs COCO-Object +3.3, no extrapolation);
(c) prereg amendment (fresh alphabetical SCLIP search) disclosed in-text.
R2 (REVA): significance 7->8, Weak Accept -> borderline Accept. Required fixes
applied: (a) "matches" -> "consistent with" + single-run/single-seed qualifier;
(b) W16-D attribution corrected -- negative expansion (either kind) reaches
official parity, credit not given to VABS selection; NULL repositioned as third
independent confirmation of the gap-vs-recovery boundary; (c) Limitations now
state SAM arbitration is not validated on any official release.

## W16-E SAM arbitration on official SC-CLIP logits: FULL-REVA TRANSFER PASSES (prereg_w16e_sam_authorcode.md)
Official SC-CLIP forward untouched through its postprocessed probability map;
SAM ViT-B pps16 region-mean pooling applied as additive layer, repo's own
prob_thd=0.15 re-applied identically; full VOC val 1449 (w16e_*.json):
- plain:      pix 42.97 (regression: exact W16-C match) / +SAM 44.10 (+1.1)
- plain+VABS: pix 59.62 (regression: exact) / +SAM 63.16 (= full REVA)
- plain+rand: pix 56.53 / +SAM 58.96
Frozen criteria: SAM transfer (4)-(2) = +3.5 >= +1.0 PASS; selection
advantage under SAM (4)-(5) = +4.2 > 0 PASS. Full REVA on official logits
recovers 93% of the author-code naming-engineering gap (63.16 vs engineered
64.60 from plain 42.97). Scope: arbitration runs on the official probability
output, not inside the repo's postprocess; SAM config is REVA's frozen one.

## W16-E incremental review (R2): Accept (from Weak Accept/borderline)
Numbers all verified. One required fix applied: the 93% gap-closure denominator
is the PIXEL-LEVEL official reference (64.6), not compute-matched -- qualifier
added in the anchor paragraph and abstract to distinguish from the 85-96%
in-stack matched-compute closures. Rebuttal A2 fully closed (base fragility,
VABS gain, full-REVA gain all author-code-anchored). Remaining non-blocking:
hand-list cross-vocab (partially addressed by W14-N3), Trident/TCC absence.

## W17 official Trident anchor: ALL FROZEN CRITERIA PASS (prereg_w17_trident_anchor.md)
Unmodified official Trident (github.com/YuHengsss/Trident), ViT-B/16 + DINO +
SAM ViT-B, --sam_refine, full VOC val 1449; only the class-name file changed
(repo's '; ' alias convention); dedicated env torch 2.1.0/mmcv 2.1.0
(w17_trident_*.log):
- official names: 67.09 (published 67.1 -- exact reproduction)
- plain:          47.44  -> NEG +19.65
- syn100:         41.69  -> synonym drop 5.75
- plain+VABS64:   63.48  -> +16.0 over plain (criterion >= +3 PASS),
                            recovers 81.6% of the author-code NEG gap
- plain+rand64:   60.71  -> selection advantage +2.77 > 0 PASS
Reading: 4th author-code anchor, first from the SAM-coupled family; audit axes
replicate (NEG +19.7 close to our-protocol scale); VABS adds value INSIDE a
SAM-based pipeline (no absorption, unlike LPOSS label propagation); no repo
knob changed. Engineering: new conda env 'trident' (repo requirements); scclip
env failed on torch-1.10 dtype incompatibility (documented, no code patched).

## W17 incremental reviews: both papers hold/improve
Audit (R1): numbers verified; soundness 8->8.5, Accept maintained. Required
fix applied: leaderboard caption now states anchors are separate experiments
(4 methods named), not table rows; added "Trident's own SAM refinement does
not diminish NEG -- spatial refinement corrects boundaries, not naming".
REVA (R2): numbers verified; Accept maintained (more stable). Required fix
applied: "covered as a host" narrowed to VABS-component-only (pixel folding,
Trident's SAM refinement as shipped, our arbitration not applied there).
Remaining non-blocking: A1 hand-list differentiation (partially covered by
W14-N3), human-sourced vocabularies (needs humans).

## W17-B official Trident anchor on COCO-Object (prereg_w17b_trident_cocoobj.md)
Unmodified official Trident, cfg_coco_object.py, --sam_refine, full val 5000
(same converted dataset as W16-D), only name file changed (w17b_trident_*.log):
- official: 41.10 (published 41.1 -- exact reproduction)
- plain:    39.08  -> NEG +2.02 (10x smaller than Trident's VOC +19.7)
- syn100:   29.01  -> synonym drop 10.07 (much larger than its VOC 5.8)
- VABS-64:  40.36  -> +1.28 over plain: VABS transfer criterion (>=+3) FAILS
- rand-64:  40.43  -> selection advantage -0.07: FAILS (NULL retained)
Frozen boundary prediction (stated before running) was confirmed: small
author-code NEG (+2.0) -> VABS ~ random, selection NULL. This is the 4th
independent confirmation of the gap-vs-recovery boundary and the 2nd on
author code (after W16-D SC-CLIP: NEG +3.3, VABS +3.5 ~ rand +3.6). Note the
axes decouple: COCO-Object naming-engineering headroom is small, but synonym
fragility is LARGER than on VOC on the same author code -- background
headroom and naming fragility are different quantities.

## W18 ANS on official Trident code: AUTHOR-CODE WORST-CASE BOUND STANDS (prereg_w18_ans_authorcode.md)
Frozen ANS protocol run end-to-end on unmodified official Trident forward
(vocab swaps recompute query features exactly as repo __init__; frozen
subsets: sorted val 0-99 search / 100-299 heldout; candidate pools from
Trident's own CLIP, WordNet cosine [0.70,0.95]; greedy alphabetical one
pass). w18_ans_trident.{json,log}:
- search-100 final: 20.37
- heldout-200: ANS 18.20 vs plain 47.84 vs syn100 40.68
ANS is 22.5 mIoU below syn100 (frozen criterion: >=5 below -- PASS
decisively) and 29.6 below plain. The searched worst-case axis is NOT
protocol-conditional: it replicates on author code with the authors' own
forward, on the SAM-coupled family. Last audit caveat ("ANS figures are
our-protocol-only") now closes; single alphabetical search, search-order
control not re-run on author code (disclosed per prereg).

## W18 incremental review (audit R1): Accept maintained, soundness 8.5
Numbers verified (22.5 margin = 4x criterion; no search->heldout shrinkage
noted as a strength). Required fix applied: "not an artefact of our protocol
either" narrowed to existence claim ("replicates end-to-end in an unmodified
official pipeline ... demonstrated on one official codebase"); added
non-comparability note between Trident ANS 18.2 and in-protocol 13.4/13.7.
Reviewer's remaining open item "LPOSS official anchor" is a reviewer memory
slip -- W15 LPOSS anchor exists and is in the paper; no action.

## W17-C official Trident anchor on Context-60 (prereg_w17c_trident_context60.md)
Unmodified official Trident, cfg_context60.py, --sam_refine, full Context val
(5105 images; our converted 60-class GT from the full .mat annotations, same
mapping as our stack's ctx60; conversion script cvt_ctx60_val.py), only name
file changed (w17c_trident_*.log):
- official: 40.07 (published 38.6; diff +1.47, within frozen 1.5 tolerance;
  we are ABOVE published -- our GT conversion may differ slightly from the
  authors' Detail-API labels; disclosed)
- plain:    39.92  -> NEG +0.15 (essentially zero naming-engineering headroom)
- syn100:   27.54  -> synonym drop 12.38 (largest observed on author code)
- VABS-64:  36.51  -> -3.41 vs plain: VABS gain criterion FAILS, and the
  effect is NEGATIVE -- at zero headroom, negative expansion actively harms
- rand-64:  35.83  -> VABS-rand +0.68 > 0, but both arms are harmful; no
  benefit claim is made from this comparison
Boundary model sharpened on author code: VOC (NEG +19.7) -> large VABS gain
+16.0 and positive selection; COCO-Object (NEG +2.0) -> VABS ~ rand, small
gains; Context-60 (NEG +0.2) -> negative expansion HARMS (-3.4/-4.1). This
matches the disclosed in-stack ADE null-to-harmful boundary and upgrades
"nothing to recover" to "expansion can hurt" at the zero-headroom end.

## W17-C incremental reviews: both maintain Accept; required fixes applied
Audit R1 (soundness 8.5 maintained): (a) "monotone decoupling" replaced by
"dissociate (n=3, observation not law), ordered by NEG size the synonym drop
runs opposite"; (b) added internal-consistency sentence (all arms share the
same converted GT, conversion shifts absolute levels only).
REVA R2 (Accept maintained; clarity 7->8 conditional on warning): Limitations
upgraded to explicit Applicability warning with the author-code dose curve,
a vocabulary-composition heuristic (thing-centric + residual background vs
stuff-inclusive) flagged as post-hoc observation, and deploy-time
applicability testing declared an open problem (user cannot measure NEG
without an engineered vocabulary).

## P3-E1 OVOD go/no-go: GO (prereg_p3_e1_ovod.md, frozen before runs)
Native COCO detection AP (pycocotools, val2017 first 1000 ids), only names
change (p3e1_owl_*.json, p3e1_yw_*.json):
- OWLv2-base:  plain 48.4 / syn1st 36.2 / syn2nd 31.1 / dis+200 47.2
  -> mean synonym relative drop 30.6% (>=15% criterion: PASS)
- YOLO-World-L: plain 46.8 / syn1st 35.3 / syn2nd 30.0 / dis+200 44.1
  -> mean synonym relative drop 30.2% (PASS)
GO criterion (either model >=15%) passed by BOTH models at 2x margin.
Descriptive: distractor+200 barely moves AP (-2.5%/-5.7%) -- native
detection AP is nearly distractor-robust (per-query decoding, no shared
softmax/background absorption), a sharp contrast to segmentation's
collapse-to-4; this strengthens the "metric convention & background
absorption" mechanism story and gives the third paper its own phenomenon
space. Precondition C1 of the paper-3 evaluation card is satisfied; C2
(E3 detection-specific coupling) remains before formal commitment.
Disclosures: 1000-image subset; syn2nd = second-ranked synonym rule
(deterministic generator has no seed at 100%; prereg amended before runs);
OWLv2 threshold 0.1 / YW conf 0.001 defaults identical across arms.

## P3-E3 detection-specific coupling: NO-GO (prereg_p3_e3_coupling.md)
OWLv2, COCO val2017 first 1000 (p3e3_coupling.json):
- coexistence (plain80+syn80): mAP 47.7 vs plain 48.4 (-0.7, < 3 AP bar)
- cross-name merged: 47.5 (delta -0.2, < 3 AP bar)
- duplicate rate 0.23% (<< 15% bar)
Both frozen C2 thresholds missed -> per the evaluation card, the third
paper does NOT stand as an independent line; the OVOD findings degrade to
an audit-paper extension. E3b decomposition (descriptive): synonym AP drop
is recall-loss-driven -- GT recall@0.5 falls 78.5% -> 58.1% (21.5% of GT
lost outright, only 1.2% newly gained); score reordering is secondary.
Combined P3 verdict: C1 GO + C2 NO-GO = no independent paper; E1/E3
results are folded into the audit paper's cross-family section (native
detection AP: synonym-fragile ~30%, distractor-robust, recall-loss-driven)
where they sharpen the metric-convention/background-absorption mechanism
contrast. Evaluation card, preregs, runs archived; line closed honestly.

## W19 RF-CLIP (AAAI 2026) author-code anchor (prereg_w19_rfclip.md)
Unmodified official RF-CLIP, trident env (torch 2.1; scclip env torch 1.10
lacks average_attn_weights), only class-name files changed:
- VOC-21: official 64.74 (pub 64.8, PASS) / plain 40.97 / syn100 36.83
  -> NEG +23.8 (largest across all five anchors), syn drop 4.1
- COCO-Object: official 36.79 (pub 37.9, d=1.11 PASS) / plain 31.86 /
  syn100 25.89 -> NEG +4.9, syn drop 6.0
- Context-60: official 37.97 vs pub 36.4, d=1.57 > frozen 1.5 gate ->
  FAILED REPRODUCTION (our converted GT, as in W17-C, likely contributes);
  per prereg no interpretation; descriptive only: plain 37.73, syn100 24.91.
Both frozen expectations hold on the two passing datasets: NEG > 0,
syn100 < plain. RF-CLIP is the fifth author-code anchor, the first 2026
method covered, and extends the anchor family to the attention-surgery
lineage's newest member; NEG +23.8 exceeds SC-CLIP's +21.6.
Engineering disclosures: (1) OOM in official postprocess with 145-query
alias one-hot (same as W16-D SC-CLIP); replaced with mathematically
equivalent per-class chunked max, regression-verified VOC official 64.74
bit-identical (w19_regress_voc.log). (2) A mid-run class-file contamination
(concurrent regression run overlapped the queue; detected because official
and syn100 COCO-Object arms returned identical mIoU 25.89) was fixed by
restoring the repo name file from GitHub and rerunning both arms cleanly;
final numbers above are from the clean rerun. (3) SegmentationClassAug
symlinked to SegmentationClass (val labels identical; dataset-side only).

## W20 CorrCLIP (ICCV 2025 Oral) author-code anchor (prereg_w20_corrclip.md)
Unmodified official CorrCLIP (metaclip_fullcc ViT-B-16-quickgelu + dino_vitb8
+ authors' pre-generated SAM2 region masks, mask_generator=None), trident
env, only class-name files changed. Masks are vocabulary-independent, so
all arms share identical region proposals.
- VOC-21: official 74.79 (pub 74.8, PASS) / plain 54.30 / syn100 48.50
  -> NEG +20.5, syn drop 5.8
- COCO-Object: official 43.67 (pub 43.7, PASS) / plain 42.26 / syn100 32.51
  -> NEG +1.4, syn drop 9.8
- Context-60: official 45.71 vs pub 44.2, d=1.51 > frozen 1.5 gate ->
  FAILED REPRODUCTION (converted GT as in W17-C/W19); descriptive only:
  plain 45.36, syn100 30.20 (drop 15.2).
Sixth author-code anchor; first mask-generator family (SAM2+MetaCLIP+DINO)
and the training-free performance-ceiling representative. The NEG/synonym
dissociation replicates the Trident ordering on a second SAM-family
method: NEG +20.5 / +1.4 / (+0.35 descriptive) vs syn drops 5.8 / 9.8 /
(15.2 descriptive) run opposite. Frozen expectations hold on both passing
datasets. Region masks identical across arms -> all deltas are text/name
interface effects by construction.
Engineering disclosures: (1) metaclip and dino weights fetched locally and
copied into the experiment machine's caches (HF/fbaipublicfiles slow or
unreachable there); no code change. (2) The authors' Context mask set has
5104 files vs our 5105-image val list (2010_001606 absent); a 5104-image
val list was used for all three Context arms (dataset-side, disclosed).
(3) VOC SegmentationClassAug symlink as in W19.

## ReME anchor evaluation: NOT FEASIBLE (2026-08-03)
The third survey recommendation (ReME, ICCV 2025, data-centric retrieval)
cannot be anchored: the public repo contains only README, install.sh and
dataset-prep scripts -- no method code, no reference set, no eval pipeline.
Cited in related work as the data-centric family with an explicit note that
it was not auditable at audit time. No experiment run.

## W19/W20 incremental review (2026-08-03)
Adversarial incremental review of the RF-CLIP/CorrCLIP anchor paragraphs:
all numbers verified against runs, citations verified against AAAI OJS /
ICCV, conditional 9/10 Strong Accept maintained. All 3 must-fixes + 4
suggestions applied: (1) OOM chunked-max workaround disclosed in the
anchor paragraph with bit-identical regression note; (2) CorrCLIP Context
5104-image list disclosed in text; (3) "exactly" softened to "within 0.1";
(4) "most dependent" qualified to on-VOC/among-anchors; (5) class-file
contamination incident added to the artifacts appendix; (6) ReME wording
tightened to "released repository contains no method code"; (7) Context
failed-reproduction wording unified across both anchors.
Review archived at stage6_review/incremental_w19w20_review.md.

## W21 VABS component on official CorrCLIP (prereg_w21_corrclip_vabs.md)
Official CorrCLIP forward unmodified; VABS-64 / matched random-64 appended
to the background line via the repo's own '; ' alias mechanism. Region
masks identical across arms.
- VOC-21 (NEG +20.5): plain 54.30 -> +VABS64 69.54 (+15.2, criterion >=+3
  PASS; recovers 74% of the author-code NEG gap) ; +rand64 65.63 ->
  selection advantage +3.9 > 0 PASS.
- COCO-Object (NEG +1.4): plain 42.26 -> +VABS64 38.18 (-4.1), +rand64
  38.95 (-3.3). Frozen prediction (VABS - plain < +3 at near-zero
  headroom) HOLDS, and the regime is actively harmful, matching the
  null-to-harmful boundary seen on Trident Context-60 and in-stack ADE.
  VABS - rand = -0.8, descriptive only, no benefit claim.
Fourth official codebase on the gap-vs-recovery dose curve; second
SAM-coupled host with a large-gap gain + positive selection advantage, and
the first author-code harm point at NEG as small as +1.4. VABS-only
component evidence; no full-REVA arbitration claim on CorrCLIP.

## Collision rescan (window after 2026-08-02, run 2026-08-03)
Intel session verdict: NO-THREAT. No new OVSS vocabulary-audit papers, no
new versions/derivatives of SynCLIP/ActiveSAM/FreeCP/RF-CLIP/CorrCLIP/ReME,
no OVOD vocabulary-audit work, no negative-word/background-synthesis
plugins in window (arXiv latest announcement batch 07-31; 07-25+ margin
re-checked). Watch item: SynCLIP derivatives, ActiveSAM venue submission.

## W22 FreeCP anchor + distractor interaction: ABORTED AT FEASIBILITY GATE
See prereg_w22_freecp.md. FreeCP's official code requires precomputed
per-name Vicuna-13B description dicts keyed by the official class names;
perturbed vocabularies cannot be expressed without regenerating LLM assets
(a method-asset modification, out of protocol). Zero runs. Retained as a
bounded audit-paper note on vocabulary engineering via LLM description
assets.

## W21 incremental review (adversarial R2) + fixes (2026-08-03)
Verdict: numbers all reconciled, Accept maintained conditional on 5
must-fixes, all implemented same-day:
- M1 "first harm point at NEG as small as +1.4" was wrong twice (Trident
  Context-60 +0.2 already harmed; +1.4 is LARGER) -> rewritten as
  "extends the harmful regime up to NEG +1.4, largest headroom with harm".
- M2 "replicates both ends" -> "reproduces recovery end, extends harmful
  end", plus explicit cross-host non-monotonicity note (CorrCLIP harms at
  +1.4 while Trident ties at +2.0).
- M4 dose-curve caption now defines parenthesised selection values
  (differences between two harmful arms, descriptive only).
- M5 Trident/COCO tie wording unified to 40.36 vs 40.43, selection -0.1.
- M6 Limitations applicability warning now includes the CorrCLIP COCO
  +1.4 harm point.
Suggestions also done: single-run/single-seed + pre-generated-mask
 disclosure in CorrCLIP paragraph; abstract updated to four codebases.

## W22 addendum: audit-paper FreeCP note + citation fixes
FreeCP feasibility-abort finding written into audit related work (closed
over engineered names via precomputed Vicuna-13B description dicts; not
auditable under only-change-name-files protocol); freecp2025 bibitem added
to audit paper and corrected to ICCV 2025 in REVA. Both papers recompiled
with zero undefined references.

## W23 REVA arbitration redundancy test on official CorrCLIP (prereg_w23_corrclip_arbitration.md)
Additive SAM ViT-B arbitration (exact W16-E mechanics) on official
CorrCLIP probabilities, VOC-21 full val, compared against the repo's own
final postprocess (which already mode-votes over its SAM2 masks):
- VABS arm: official 69.54 (bit-matches W21 -> regression PASS) -> ours
  68.25. Arbitration gain -1.29: frozen criterion (>= +1.0) FAILS, matching
  the frozen honest prediction (redundancy expected on an already
  region-arbitrated host).
- rand arm: official 65.63 (bit-matches W21) -> ours 64.05 (-1.58).
  Selection under arbitration +4.19 (descriptive only).
Verdict per prereg: scope boundary, not transfer. Our arbitration layer
adds value on pixel-level hosts (SC-CLIP +3.5, W16-E) but is redundant /
slightly harmful on hosts that already arbitrate over regions --- the
REVA mechanism claim sharpens: the gain comes from region evidence,
however injected; stacking a second region vote does not help. Limitations
keeps "full REVA validated on one official release" and gains the
redundancy-boundary sentence.
Process incident: the remote-only converted rand class files had been
deleted by a sync --delete (created remotely in W21, never copied back);
recreated deterministically from the same source files + same sed
transform (official arm bit-match confirms equivalence), rand arm rerun.

## W22/W23 incremental review + fixes (2026-08-03)
Reviewer verdicts: FreeCP note KEEP (1 verification: confirmed by code
inspection that query_featuresLLM is consumed unconditionally in the
purification step -- no official non-LLM mode exists; wording anchored to
"in its official configuration" + "at audit time"). W23 paragraph REVISE,
all done: added VOC/single-run/>=+1.0-criterion disclosure, added the rand
arm (65.6->64.1), downgraded the host-level qualitative claim to a
single-dataset boundary observation, resolved the surface contradiction
with the preceding "not run inside any official postprocess" sentence
(now "not run as a replacement ... additive test on one (CorrCLIP)").
Both papers recompiled, zero undefined references.

## W24 SNB: background subspace geometry + angle applicability criterion (prereg_w24_snb.md) — H1 FALSIFIED, H2 MARGINAL
Ideator round-6 mechanism-upgrade candidate (intel verdict SAFE: no prior
text-side background subspace work). SCLIP ViT-B/16 text encoder throughout.
- H2 (theta-Spearman, run FIRST per prereg, near-zero compute): mean
  principal angle between plain-vocabulary span and the top-r background
  basis (SVD of the frozen 1633-word lexicon embeddings) vs 13 archived
  oracle VABS-minus-plain gains (4 distinct vocabularies, hosts share
  theta). Frozen primary r=32: Spearman rho = 0.685 — BELOW the frozen 0.7
  PASS bar, above the 0.4 falsification line -> MARGINAL, no predictor
  claim. Post-hoc r-ablation (descriptive only, not usable as evidence):
  r=8/16 give rho 0.796 (LOO 0.767..0.855); r=64 degrades to 0.435.
  Direction is right (theta orders VOC >> COCO ~ Context > ADE, matching
  the gain ordering at dataset level) but the frozen criterion did not
  pass. File: runs/w24_theta.{json,log}.
- H1 (SNB projection-norm background channel vs discrete VABS-64,
  VOC-21 test-300): FALSIFIED decisively. SCLIP: plain 34.75 / VABS64
  53.62 / SNB(r=32, per-image mean-calibrated) 34.53. NACLIP: 36.54 /
  52.69 / 34.36. SNB ~= plain on SCLIP and 2.2 BELOW plain on NACLIP —
  the continuous projection
  norm, calibrated to the plain background-word scale, is nearly uniform
  across patches and never wins the argmax where it matters; IN THIS
  FROZEN INSTANTIATION (global SVD basis, projection-norm score,
  mean-matched calibration) the discrete max-over-negatives channel
  carries per-patch discriminative structure the subspace-energy scalar
  does not — other subspace scorers were not tested and the negative
  result is limited to this instantiation. SNB < VABS64 - 1.0 -> kill per
  prereg. File: runs/w24_snb.{json,log}.
Verdict: the "background as text-embedding subspace" upgrade FAILS in its
frozen form; preserved as a negative result. The angle diagnostic is
suggestive but unproven (marginal at frozen r; only 4 distinct vocab
points). Fallback per ideation plan: image-side/prediction-side dosage
signals (thermostat, independent signal source) remain unexplored.

## W25 NEG-Thermostat H1/H2 pilot (prereg_w25_thermostat.md) — H1 PASS (S1), H2 switch FAIL under frozen rule
Label-free deployment-time signals vs 8 archived oracle VABS-64 gains
(SCLIP/NACLIP x voc21/cocoobj/ctx60/ade150, first 50 unlabeled images,
plain vocab, GT never read). Frozen criterion: >= 1 signal with
|Spearman| >= 0.6 in its PRE-DECLARED sign.
- S1 crowding (mean max cos-sim to queries, declared NEGATIVE):
  rho = -0.714 as first computed -> PASS. The only correct-signed
  passing signal. ERRATUM (R2 incremental review, 2026-08-04): the
  frozen GAINS table transcribed the ctx60/ade150 oracle gains from the
  rounded RESULTS summary line with the host assignment ambiguous; the
  primary-source values (W13-L5t: sclip/ctx60 +0.62, naclip/ctx60
  +0.52, sclip/ade150 -0.97, naclip/ade150 -0.65) give rho = -0.690 --
  STILL past the frozen 0.6 bar with the pre-declared sign, so H1 PASS
  stands, but the margin is thin and the pass is reported as fragile
  (a two-rank perturbation moves rho by 0.02-0.07). Also disclosed:
  ctx60/ade150 gains are GT-present-mIoU protocol while voc21/cocoobj
  gains are standard-mIoU (rank correlation only, but protocols mixed).
- S2 entropy (declared POSITIVE): rho = -0.833 -> WRONG SIGN, does not
  count. Honest finding: entropy anti-correlates with headroom, likely
  confounded by vocabulary size (21 vs 150 classes); prediction
  uncertainty is NOT a headroom proxy.
- S3 background-basis energy: rho = -0.024 -> NULL (consistent with the
  W24 subspace failure).
H2 pilot (frozen global LOO midpoint switch on S1, k in {0,64}):
retained helpful gain 98.0% (bar 90% PASS) BUT do-no-harm FAILS --
sclip/ade150 (gain -0.97 corrected) classified ON (loss 0.97 > 0.2 bar): S1 scales
are host-dependent (naclip offsets +0.01..0.02) so one global threshold
cannot separate; per-host thresholds would separate cleanly at the cost
of always switching off ctx60 (+0.5/+0.6 forgone) -- descriptive only,
not frozen, no claim. Verdict: a deployment-time headroom SIGNAL exists
(crowding, preregistered sign, 8 points, effective n closer to 4 since
hosts share datasets — caveat frozen in prereg), but no validated
switch rule yet; any future rule needs per-host calibration frozen in
advance and the official-code anchors as out-of-sample test. Files:
runs/w25_thermostat.{json,log}.

## W26 crowding out-of-sample ordinal test on official codebases (prereg_w26_crowding_oos.md) — MARGINAL (4/5 pairs)
Read-only crowding margin M = mean[log p_max - mean_c log p_c] from each
official repo's stored per-class map (plain name file, first 50 val
images, GT never read; frozen primary = M, max-prob descriptive;
class-count confound disclosed in advance). Within-host prediction:
higher M => lower oracle VABS gain.
- corrclip: M voc 2.47 < coco 3.45, gains +15.2 > -4.1 -> pair CORRECT.
- trident: M voc 2.54 < coco 3.48 < ctx 3.88, gains +16.0 > +1.3 > -3.4
  -> all 3 pairs CORRECT (including the harmful ctx60 endpoint).
- scclip: M voc 17.21 > coco 5.02 with gains +16.7 > +3.5 -> pair WRONG.
  scclip's stored map is post-PAMR/alias-max (much deeper postprocess
  than the other two repos; its VOC margin is an order-of-magnitude
  outlier), a frozen-in-advance disclosed heterogeneity.
Frozen verdict: exactly 1 of 5 pairs wrong = MARGINAL -> reported, NO
validated-signal claim. Both hosts whose stored maps are plain
softmax/background-fold (trident, corrclip) order perfectly, including
both harmful endpoints; the failure comes from the host with the
deepest stored postprocess. Suggestive but unproven; any future claim
needs matched extraction depth, frozen first. Files:
runs/w26_{scclip,trident,corrclip}.{json,log}.

## W24-W26 incremental review (R2, adversarial) + collision scan 2026-08-04 — verdicts held after erratum
R2 recomputed all statistics from archived JSONs: W24 verdicts held; W26
held (also found max-prob variant would give only 2/5 — M was the frozen
primary, procedurally clean). BLOCKING finding M2: the W25 GAINS table
transcribed ctx60/ade150 gains from a rounded summary line with hosts
swapped; primary source (W13-L5t) resolved in favour of the
w24_theta.json assignment (sclip/ade -0.97, naclip/ade -0.65; sclip/ctx
+0.62, naclip/ctx +0.52); corrected rho = -0.690, still past the 0.6
bar with pre-declared sign -> W25 H1 PASS stands but is reported as
fragile (erratum recorded above; paper cites -0.69 with the effective-n
caveat). All must-fixes (M1 effective-n in paper, M3 NACLIP -2.18
wording, M4 instantiation-limited W24 interpretation) and suggestions
(S2 subgroup emphasis removed, S3 n=5 pairs) applied; paper recompiled
with zero undefined references. Collision scan (window > 2026-08-03):
both platforms zero new entries (Monday arXiv batch not yet released);
all 5 threat surfaces CLEAR by window; rescan advised 2026-08-04 after
the batch lands. Files: stage13_reviews/r2_review_w24_w26.md,
stage11_phase3/intel_collision_20260804.md.

## W27 crowding expansion, matched extraction protocol (prereg_w27_crowding_expand.md) — H1 PASS, H2 STRATIFIED PASS, H3 FALSIFIED
18 fresh points under one protocol (raw cosine, no softmax; 50 unlabeled
imgs for S1; test-300 single-seed gains; VABS-64 re-selected per vocab):
hosts {sclip, naclip, clearclip} x {voc21, cocoobj, ctx60, ade150}/plain
+ {voc21, cocoobj}/syn100_s0. Sanity: sclip plain gains reproduce the
W13-L5t primary-source values exactly (18.87 / 9.24 / 0.63 / -0.97),
independently confirming the W25 erratum assignment.
- H1 pooled Spearman = -0.721 (bar <= -0.6) -> PASS.
- H2 per-host Spearman = -0.829 for ALL three hosts (bar <= -0.5) ->
  STRATIFIED PASS. The host-scale confound that broke the W25 global
  threshold switch does NOT break within-host ordering.
- H3 vocabulary axis: 0/6 plain-vs-syn100 pairs correct -> FALSIFIED.
  In every pair the syn100 vocabulary has LOWER crowding AND LOWER
  gain — the opposite of the frozen prediction. Synonyms weaken the
  image-text alignment (lower cosines) at the same time as they shrink
  the recoverable headroom, so the signal cannot see naming-induced
  headroom changes. Verdict: crowding is a per-host-calibrated
  DATASET-LEVEL headroom proxy, not a vocabulary-level one; the
  "signal tracks vocabulary headroom" claim is falsified at the
  within-dataset level and the paper says so.
Caveats (frozen): single seed, test-300, one synonym draw, three hosts
share the CLIP backbone. Files: runs/w27_crowding.{json,log},
prereg_w27_crowding_expand.md; new vocab files
{voc21,cocoobj}_syn100_s0_vabs64.json (+meta).

## Collision rescan 2026-08-03 (second pass) — CLEAR, Monday batch still pending
Intel re-ran the scan same-day: arXiv latest entries still 07-28 (Monday
batch lands ~00:00 UTC 08-04), OpenReview zero new; all 5 threat
surfaces CLEAR by window. Next rescan after the batch lands.
File: stage11_phase3/intel_collision_20260803_rerun.md.

## W27 incremental review (R2) — verdicts held, 2 wording must-fixes applied
R2 recomputed all W27 statistics: pooled -0.721, per-host -0.8286 x3,
H3 0/6, sanity cross-check confirmed (also independently confirms the
W25 erratum direction). Key finding: the identical per-host rho is not
coincidence or bug — the three hosts produce IDENTICAL rank orderings
of both S1 and gain (shared CLIP backbone), so H2 is one within-host
test replicated three times, not three independent replications (M1:
qualifier added to the paper sentence). M2: "cannot see naming-induced
headroom changes" limited to synonym substitution, one draw tested.
Both fixes applied; tier verdict unchanged. File:
stage13_reviews/r2_review_w27.md.

## W28 crowding on ViT-L/14 (prereg_w28_crowding_vitl.md) — H1 NOT PASSED (one host under bar; ordering degenerate with ViT-B), H2 descriptive 1/4
12 points, sclip/naclip on ViT-L-14-quickgelu, same protocol as W27
(raw cosine S1, test-300 gains, VABS-64 re-selected with the ViT-L text
encoder; new vocab files *_vabs64_vitl.json).
- H1: naclip rho = -0.886 (passes -0.5 bar); sclip rho = -0.486
  (misses the bar by 0.014). Frozen rule required BOTH hosts -> no
  independent-replication pass. Additionally, the S1 orderings on
  BOTH ViT-L hosts are IDENTICAL to the ViT-B ordering (voc_syn <
  voc < coco_syn < coco < ade < ctx), so even a passing rho would
  have been PASS-BUT-DEGENERATE by the frozen wording: backbone
  independence remains unestablished.
- Contributing factor (honest): on sclip/ViT-L the VABS gains
  themselves are compressed (VOC +1.8 vs +18.9 on ViT-B; consistent
  with W1a's mixed ViT-L results), so the gain ranking is noisy at
  test-300 resolution.
- H2 (descriptive only, no bar per prereg): 1/4 pairs correct —
  consistent with W27's vocabulary-axis falsification; the single
  "correct" pair (naclip/voc21) differs by only 0.17pt, test-300
  noise level, and must not be read as a partial revival of the
  vocabulary axis on ViT-L.
Verdict: the applicability-signal line stays a PILOT (dataset-level,
per-host-calibrated, backbone-independence not shown); the paper
sentence updated accordingly. Files: runs/w28_crowding_vitl.{json,log},
runs/w28_vabsgen.log, prereg_w28_crowding_vitl.md.

## W29-A synonym-draw robustness of the W27 vocabulary-axis falsification (prereg_w29_syn_draws.md + Amendment A) — DRAW-ROBUST, 12/12
Feasibility finding first (zero-run): the frozen syn100 rule is
DETERMINISTIC (frac=1.0 top-cosine WordNet synonym) — syn100_s0/s1/s2
are byte-identical, so "draws" of the existing rule do not exist and
the original W29 design was vacuous; amended BEFORE any evaluation to
a new frozen rule (uniformly random second-tier valid synonym, seeds
1/2, files {voc21,cocoobj}_synr_s{1,2}.json + synr_manifest.json).
Result: ALL 12/12 plain-vs-synr pairs (3 hosts x 2 datasets x 2 draws,
ViT-B, matched W27 protocol, VABS-64 re-selected per vocabulary) show
the inverted pattern — synonym vocabulary has LOWER crowding AND LOWER
gain. The W27 vocabulary-axis falsification is draw-robust and extends
to harder second-tier synonyms (caveat: both draws share the WordNet
synonym source lexicon, so they are not fully independent
perturbations); the paper caveat "one draw tested" is
replaced by the 12/12 replication sentence. No new positive claim is
created. Files: runs/w29_synr.{json,log}, runs/w29_gen.log.

## W28/W29-A incremental review (R2) — verdicts held, 2 wording must-fixes applied
R2 recomputed all W28/W29 statistics (identical); confirmed the W28
"not passed vs falsified" distinction, the degenerate-ordering
self-disclosure, and the legality of the W29 Amendment (frozen before
any evidence generation, criterion untouched). Noted: at n=6 the
Spearman grid steps by 1/35, so the -0.5 bar is effectively -0.514 and
sclip missed by the minimum possible increment (bar executed as frozen).
M1 (W28 H2 1/4 noise-level qualifier) and M2 (byte-identity scope
limited to voc21; cocoobj determinism is rule-derived) applied; S5
(shared-lexicon caveat) added. Tier verdict unchanged. File:
stage13_reviews/r2_review_w28_w29.md.

## Collision scan 2026-08-04 (Monday batch landed) — 2 WATCH, 0 THREAT
345 new/updated cs.CV entries (>=08-01) filtered. Faces 1 & 4 raised to
WATCH by OVEarth-Bench v2 (arXiv:2607.27278, EO domain): category
breadth + query diversity within a FIXED benchmark vocabulary
(negatives = absent classes to reject); it does NOT perturb the names
of a fixed target class — orthogonal to our naming-sensitivity audit
(intel verified "synonym" appears only in its appendix category
alignment). Cited in the audit paper related work with the
self-distinction sentence. Faces 2/3/5 CLEAR; SynCLIP/ActiveSAM/all
anchored methods: no new versions; crowding/applicability niche still
empty. Next-scan window should start 08-03 (batch tail indexing).
File: stage11_phase3/intel_collision_20260804.md.

## W30 text-side predictability of per-class synonym damage (prereg_w30_textpred.md) — NULL (both bars fail; negative result preserved)
Motivation: text-space dual of CorrCLIP's scope-value decomposition
(scoop-check verdict RISK -> scope frozen to PREDICTABILITY only;
stage11_phase3/intel_scoop_textcorr_20260804.md). 12 cells re-run with
per-class dumps: {sclip,naclip,clearclip} x {voc21,cocoobj} x
{plain,syn100_s0}, test-300 (run: w30_textpred.json/.log).
- H1 (P1 = cos(orig,syn) vs per-class IoU delta, COCO primary, bar
  >= +0.40 in >=2/3 hosts): FAIL — rho = +0.341/+0.382/+0.319 (0/3).
  Consistent positive sub-bar trend across all three hosts; descriptive
  only, and hosts share the CLIP backbone (not independent).
- H2 (P2 = change in max inter-class cosine, bar <= -0.40 in >=2/3
  hosts): FAIL decisively — rho = -0.057/-0.050/-0.042 (~zero).
  Inter-class "scope shift" carries no per-class signal at all.
- VOC descriptive (n=14): mixed (sclip P1 -0.17, P2 -0.67; others
  P1 +0.24/+0.35) — small-n noise.
Verdict per prereg: NULL. Extends W15-E — per-class synonym damage is
NOT first-order text-predictable (neither name fidelity nor competition
shift clears the bar). Third text-side per-class predictor failure
(W15-E, W24-H2 marginal, W30); the pattern is now itself evidence: the
damage mechanism is not readable from FIRST-ORDER text-embedding
statistics (higher-order/combined features untested).
Note: usable COCO n shrank 65 -> 47 because both arms require the class
present in GT on the test-300 subset (script filter matches prereg).
Independent recompute of all 12 Spearman values matched (R2 review
stage13_reviews/r2_review_w30_paper.md: W30 PASS; M1-M3 wording fixes
applied to both papers — mechanism-surface range corrected to 5.8-12.4
after R2 caught the 15.2 endpoint coming from the gate-failed
CorrCLIP Context descriptive cell).

## Collision scan 2026-08-05 (Tuesday/Wednesday batch) — 2 THREAT flags, boundary review in progress
Windows >=08-03 covered (incl. previously unindexed 08-03 tail + Wed
mailing to 08-04 17:59). Faces 1/4/5 CLEAR; OVEarth-Bench stays v2, no
general-domain expansion.
- THREAT (face 2): PTC "Perceptual Anchoring" (arXiv:2608.03991,
  08-04) — training-free plug-in that calibrates TEXT embeddings using
  prototypes built from reliable visual evidence, with calibration
  strength adaptive to evidence amount. Overlaps our "test-time
  adaptive dosage" theme (different mechanism: text calibration, not
  negative-word synthesis). Deep-read boundary check dispatched
  (3 questions: background negatives? applicability on/off signal?
  harmful-case reporting?).
- THREAT (face 3): EOVSAM (arXiv:2608.02284, 08-03 tail) — concurrent
  competitor to ActiveSAM's niche (SAM3 vocabulary-traversal cost);
  trained one-pass aggregation vs ActiveSAM's training-free pruning.
  ActiveSAM is a TRACKED EXTERNAL paper, not ours — no action needed
  for our two papers; recorded for ecosystem awareness only.
File: stage11_phase3/intel_collision_20260805.md.

## PTC boundary review (2026-08-05) — THREAT downgraded to WATCH; cited in REVA related work
Deep read of PTC (arXiv:2608.03991; Ma, Lu, Peng, You) confirms
orthogonality: (1) no background negatives/channels — calibrates text
embeddings of GIVEN classes with visual prototypes (Sec 3.1/Eq.11);
(2) dosage is per-image x per-class hard gate + log modulation
(Eq.9-11), dataset-level dials (K_min, mu) are LABEL-tuned per dataset
per baseline (mu=0.02 on COCO-Object ~= off) — no label-free
headroom/crowding signal; (3) harmful regime observed (ViT-L
COCO-Object mIoU drop, Table 6) but NO switch-off rule. Uses
SCLIP/ProxyCLIP standard plain names; never touches official-vs-plain
provenance. Action: cited as concurrent work in REVA related work with
the orthogonality + "corroborates low-headroom boundary" sentence;
recompiled clean. EOVSAM (2608.02284) affects tracked-external
ActiveSAM only — no action for our papers.
File: stage11_phase3/intel_ptc_review_20260805.md.

## W31 per-patch negative aggregation (prereg_w31_negagg.md) — H1 NULL (no component), H2 PASS (mechanism curve), D1 supports per-patch selectivity
Scoop check SAFE (stage11_phase3/intel_scoop_perpatch_neg_20260806.md).
3 hosts x {voc21, cocoobj}, test-300, archived VABS-64 vocabs; background
channel = mean of top-k negative sub-query probs, k in {1,2,4,8,16,64}
(run: w31_negagg.json/.log).
- H1 (some k in {2,4,8} beats amax by >=+0.5 on VOC in >=2/3 hosts):
  FAIL — k=1 (amax) is strictly optimal everywhere; every k>1 loses
  monotonically (e.g. sclip VOC 53.62 > 52.75 > 50.71 > 47.04 > 40.46
  > 27.24). NULL for the derived component.
- H2 (monotone degradation toward k=64; k=64 at least 3 below amax on
  VOC in all hosts): PASS decisively — k=64 (mean = discrete analogue of
  W24's energy scalar) collapses to 27.2/29.0/28.6, i.e. 23-26 mIoU
  below amax, and BELOW plain (34.75/36.54): averaging over the negative
  set destroys the background channel exactly as the W24 interpretation
  predicted. The k-curve is the missing bridge between discrete VABS and
  the falsified SNB subspace.
- D1 (winning-negative concentration, 50 VOC imgs/host): top-8 negatives
  cover only 63.6%/58.2%/58.8% of background-win patches (<90% bar in
  all hosts) — per-patch selectivity is real, not an illusion of a small
  global subset.
Regression: k=1 VOC values match W24 vabs64 arm bit-for-bit
(53.62/52.69). Verdict per prereg: component NULL; mechanism evidence
(H2+D1) eligible for the REVA mechanism section: the background channel
needs per-patch discrete competition — max over a diverse negative set —
and any energy-like aggregation (continuous subspace or top-k mean)
degrades monotonically with the amount of averaging.
R2 incremental review (stage13_reviews/r2_review_w31.md): all recomputes
bit-for-bit; verdicts upheld. Fixes applied: 52.8->52.7 rounding; D1
scoped to VOC/50-img; "genuinely patch-dependent" weakened — R2 found a
global top-24 subset covers 91.8-93.2% of wins in all hosts, so a
pruned-global-set variant (amax over global top-k) is NOT excluded and
flagged as untested (candidate future prereg).

## W32 pruned global negative set (prereg_w32_prunedneg.md) — H1 FAIL (pruning loses; diversity claim strengthened), H2 PASS (win statistics beat coverage selection at matched budget)
Follow-up to R2's W31 finding (global top-24 covers ~92% of wins).
3 hosts x {voc21, cocoobj}, test-300, amax arms (run: w32_prunedneg.json):
- A0 full 64 / A1 bg+top-24 winners / A2 bg+facility-location-24:
  sclip voc 53.62/52.62/49.29; naclip voc 52.69/52.11/48.78; clearclip
  voc 51.37/50.91/46.55; sclip coco 31.81/30.91/30.35; naclip coco
  32.81/31.69/30.88; clearclip coco 32.99/32.28/30.86.
- H1 (A1 within 0.5 of A0 everywhere): FAIL — VOC deltas 1.00/0.58/0.46
  (2 of 3 hosts over), COCO deltas 0.71-1.12 (all over). Despite ~92%
  win coverage, pruning to 24 words loses 0.5-1.1 mIoU: rank-2
  competition matters beyond win coverage. The W31 "diverse set" claim
  is now positively supported, and the paper's "pruned variant untested"
  caveat can be replaced by this measured result.
- H2 (A1 > A2 by >=0.5 on VOC in >=2/3 hosts): PASS 3/3 (+3.3/+3.3/+4.4)
  — at matched budget M=24, usage-statistics selection (50 unlabelled
  images) clearly beats facility-location coverage. Candidate cheap
  image-corpus-conditioned selection signal; caveat: the 50 stat images
  are the first 50 of test-300 (unlabelled use, GT untouched, disclosed
  in prereg), so a clean-split replication is required before any paper
  claim beyond a footnote.
Regression: A0 matches W31 k=1 bit-for-bit. Single seed, shared backbone.

## W33 clean-split replication of win-statistics selection (prereg_w33_winstat_clean.md) — H1 PASS 3/3; W32 H2 not an overlap artifact
Statistics images samples[300:350], disjoint from evaluated test-300
(run: w33_winstat_clean.json):
- VOC A1'(clean top-24) vs A2(FL-24): 53.10/49.29, 51.87/48.78,
  50.63/46.55 -> +3.81/+3.09/+4.08, H1 PASS 3/3 (bar +0.5 in 2/3).
- H2 stability: |A1'-A1_W32| = 0.48/0.24/0.28 on VOC, all within 0.5 —
  the signal is insensitive to which 50 unlabelled images are used.
- COCO secondary: A1' also beats A2 in all hosts (+1.09/+1.31/+1.95).
- A1' remains 0.2-0.8 below A0 full-64: win-statistics is a better
  SELECTOR at fixed budget M=24, not a better final method than the
  full set.
Verdict: candidate image-corpus-conditioned selection improvement,
clean-split validated at 3-host level (shared backbone caveat). Frozen
next gates before any paper/method claim: official-code anchor +
dose-curve datasets (Context/ADE harmful-regime behavior).
Scoop check (intel_scoop_corpusneg_20260806.md): SAFE — no prior work
selects negative/background words from unlabeled target-domain image
statistics; FreeCP/ActiveSAM prune foreground vocab only; CC/TCC are
text/query-conditioned; closest contrast to cite: FLOSS (unlabeled
corpus -> template selection, not negatives). Periodic scan 08-06: PTC
posted 08-04 (already cited), no OVEarth v3, EOVSAM efficiency-lane
only, per-patch negative gating still unoccupied.

## W34 win-statistics selection gates (prereg_w34_winstat_gates.md) — Gate 1 FAIL by 0.02 (one ADE cell), Gate 2 PASS (+1.53 on official ProxyCLIP); frozen rule -> NO paper claim
Gate 1 dose-curve (runs/w34_gate1.json), plain/A0/A1'/A2:
sclip ctx 26.63/27.25/27.26/26.79; sclip ade 14.10/13.13/13.00/13.32;
naclip ctx 27.46/27.98/28.04/27.59; naclip ade 15.39/14.74/14.80/14.68;
clearclip ctx 27.95/28.56/28.84/28.09; clearclip ade 15.04/14.34/14.44/14.45.
H1a: 11/12 frozen conditions pass; sclip/ade A1'=13.00 vs A2-0.3=13.02
-> miss by 0.02 (test-300 resampling noise floor is 3.1 mIoU; still a
FAIL by the frozen criterion, no reinterpretation).
Gate 2 official ProxyCLIP VOC (unmodified repo, name file swap only;
runs/proxyclip_official_plainfl24 + _plaintop24): FL-24 57.77 vs top-24
59.30 (winners transferred from OUR SCLIP stack statistics on 50
disjoint images) -> +1.53 >= +1.0, H2 PASS; NOTE: the 08-01 plainvabs
anchor (56.22) was run under a different config family and is NOT
quantitatively comparable to this batch's 57.77/59.30 — do not read
top-24 as beating full-64; the only pre-registered comparison here is
within-batch FL24 vs top24 (R2 M2 fix applied).
Verdict (frozen): a gate failed -> no paper claim; win-statistics
selection remains a RESULTS-level validated selector signal (W33 clean
3/3 + official-code transfer +1.53) with a 0.02-margin ADE boundary
miss. Any future claim requires a fresh prereg (e.g. multi-seed ADE
replication), not a reinterpretation of this one.
Env note: remote box still offline; DINO hub repo+vitb8 checkpoint
provisioned to DATA torchhub (TORCH_HOME) from the local VM.
R2 incremental review (stage13_reviews/r2_review_w32_w34.md): all
recomputes match; frozen no-claim rule verified as enforced. Fixes:
paper pruning sentence corrected to 0.58-1.12 in 5/6 cells (sixth 0.46
inside the bar) with VOC scoping on the ~92%; W34 RESULTS 56.22
juxtaposition rewritten as non-comparable config family. R2 flag for
the future: the M=24 budget lineage traces to the overlapped W31
diagnostic — decouple budget choice in any fresh prereg.

## W35 win-statistics selector v2 (prereg_w35_winstat_v2.md) — ALL FROZEN CRITERIA PASS
Derived budget M* = smallest M covering 90% of background wins on 50
unlabeled statistics images (per host x dataset x draw); 3 disjoint
draws [300:350]/[350:400]/[400:450], eval samples[:300]; matched FL
control at same M*. Run: runs/w35_winstat_v2.{json,log}.
H1 (VOC+COCO, A1*>A2*+0.5): 18/18 cells, deltas +0.52..+5.05. PASS.
H2 (ctx60+ade150, A1*>=A2*-0.3): 18/18 cells (min delta -0.25). PASS.
H3 budget stability: max within-cell ratio 13/7=1.86 < 2. Stable; M*
adapts with headroom (ADE 7-17, ctx 20-34, VOC 19-22, COCO 20-26).
vs full-64: deficit 0-1.11; on ADE beats full-64 in 6/9 cells (naclip/
clearclip all positive, up to +0.54), sclip within 0.15.
Verdict: component eligible for bounded paper subsection per frozen
rule -> added stage8 \S "A fourth component candidate: corpus-statistics
negative selection with a derived budget" (sec:winstat), outside
headline claims, with W34 gate-2 ProxyCLIP transfer (+1.5) cited and
the earlier W34 0.02-margin gate failure disclosed. Caveats retained:
single seed, shared backbone, same-domain statistics.
R2 incremental review (stage13_reviews/r2_review_w35.md): all 36 cells
recomputed and match; tolerance clause unused (results pass under the
stricter W34-style criteria too); section placement compliant. 4 fixes
applied to sec:winstat: 90% framed as frozen constant with untested
sensitivity (not "removing the free parameter"); +5.1 -> +5.0; ProxyCLIP
+1.5 annotated as the M=24 registration's selection (derived-budget
variant not re-run on official stack); H2 criteria-change vs W34
disclosed with post-hoc all-36-pass verification. Grade: maintained.

## W36 official-stack transfer of derived-budget selection (prereg_w36_anchor_mstar.md) — H1 PASS
Unmodified official ProxyCLIP, VOC full val, name-file swap only.
Selection = W35 sclip/voc21 draw[300:350] archive (M*=21, verbatim);
control = FL-21 (frozen VABS greedy prefix, verified prefix-consistent
with the archived 64-word order). Results:
FL-21 57.66 mIoU vs M*-21 59.22 -> +1.56 >= +1.0 frozen bar, PASS.
Per verdict rule, sec:winstat transfer caveat replaced by the direct
derived-budget anchor number (+1.6, alongside the fixed-24 +1.5).
Runs: runs/proxyclip_official_plainfl21, _plainmstar21, w36_anchor.log.

## Intel scan 08-07 window + fourth-component placement check — SAFE
(stage11_phase3/intel_scan_w4comp_20260807.md). Unlabelled-image-statistics
negative selection + coverage-derived budget: no prior art (all three
parts and their combination unoccupied); hard-contrast list FLOSS/CC-TCC/
FreeCP/ActiveSAM/PTC/VLOD-TTA/OVDiff. PTC v2 replaced 08-05/06 with no
mechanism change. No new OVSS methods in Wed/Thu batches; per-patch
negative gating empty for a third consecutive scan. Paper: FLOSS/FreeCP
positioning sentence added to sec:winstat (floss2025 bibitem added).
Reviewer-risk note archived: 90% coverage constant sensitivity untested
(already disclosed in the subsection).

## W37 coverage sensitivity + domain-shifted statistics (prereg_w37_coverage_domainshift.md) — ALL PASS
Runs: runs/w37_covshift.json + .log; c=0.90 cells reused from W35 archive.
Part A (coverage c in {0.80,0.85,0.90,0.95}, draw[300:350]):
H-A1 voc21 spread of A1(c): sclip 0.79, naclip 0.61, clearclip 0.61,
all <=1.0 -> 3/3 PASS. H-A2 ade150 no-harm: 9/9 new cells within
-0.3 (worst -0.29, sclip c=0.95) -> PASS. M* grows monotonically with
c (e.g. sclip voc21: 13/16/21/28) as expected.
Part B (cross-domain statistics, c=0.90): voc21 eval w/ cocoobj-image
stats: +3.51/+3.23/+4.12 vs FL control; cocoobj eval w/ voc21-image
stats: +1.82/+1.93/+1.83 -> 6/6 cells >= +0.5, H-B1 PASS (bar 2/3).
Descriptive same-vs-cross A1 gap: voc21 -0.11/+0.33/+0.36, cocoobj
-0.18/-0.70/-1.20 (cross slightly lower on cocoobj, still beats FL).
Paper: the two sec:winstat caveats ("sensitivity untested",
"domain-shifted statistics untested") replaced by measured results per
frozen verdict rule. Promotion to REVA proper still requires an
official-stack dose-curve gate (explicitly outside W37).
R2 review (stage13_reviews/r2_review_w36_w37.md): all recomputations
match; FL-21 prefix property verified at code level (greedy is
sequential by construction); c=0.90 reuse compliant; M1-M4 from the W35
review confirmed closed. Two light fixes applied: +3.24 -> +3.23 in the
Part B line above; paper cross-domain sentence now notes only
near-domain (natural-image VOC<->COCO) shift is covered. Grade: maintained.

## W38 official-stack promotion gate (prereg_w38_official_dosegate.md) — GATE FAIL (G1), promotion denied
Official ProxyCLIP, Context-60 full val (5105), name-file swap only.
ADE-150 excluded on frozen technical grounds (no background class in
the official config). Arms: P plain 36.50; A1* plain+winstat M*=34
(W35 sclip/ctx60 draw300 verbatim) 33.74; A2* plain+FL-34 prefix 34.00.
G1 (A1* >= P-0.3): 33.74 vs 36.50 -> -2.76, FAIL.
G2 (A1* >= A2*-0.3): -0.26, PASS.
Verdict per frozen rule: component stays a bounded candidate; failed
number written into sec:winstat. Interpretation (consistent with the
documented low-headroom harmful regime): on official ProxyCLIP ctx60,
ANY negative expansion harms vs plain; the selector only improves WHICH
negatives, not WHETHER to expand. Runs:
runs/proxyclip_official_ctx60{plain,mstar34,fl34}/results.txt,
w38_dosegate.log.

## 08-08 intel scan (stage11_phase3/intel_scan_20260808.md) — SAFE
Friday batch fully landed (108 new + 24 cross + 75 replace). High-priority
"when to expand negatives" unlabeled-criterion search: no direct hit,
niche still empty (nearest: TTABC 2606.14299 evidence-based TTA proxies;
CogVis 2608.06150 per-image dynamic thresholding, OVCD domain). PTC
still v2, OVEarth still v2, ActiveSAM/EOVSAM unchanged. Two new TF-OVSS
baselines to track for comparison tables (no negative mechanisms, no
collision): SCI-CLIP (2608.05627, segment-centric + retrieval memory)
and MAVISEG (2608.05878, diffusion backbone). Negative-selection /
derived-budget / per-patch gating niche empty for the 4th consecutive
scan; 4th component remains SAFE.

## 08-09 code-verification interim (intel session, ahead of 08-10 scan)
SCI-CLIP (2608.05627): official code released at
github.com/mzamini92/SCICLIP (single commit 08-05, internal name PRISM;
8-benchmark configs + reference-bank build + full eval pipeline; deps
SAM2 Hiera-L + CLIP/DINO). Meets the anchor protocol precondition ->
eligible as a comparison-table CANDIDATE, pending exact reproduction of
published values before any table entry.
MAVISEG (2608.05878): no official code (no arXiv link, no GitHub repo,
"preprint under review") -> not eligible yet; keep tracking.
Monday-batch (08-10) periodic scan report still pending from the intel
session.

## 08-10/08-11 periodic scan (intel session, Monday+Tuesday batches, report archived stage11_phase3/intel_scan_20260810.md)
Headline: TPA (2608.08290, Tue new submission) — training-free output-level
plug-in: on a small unlabeled deployment pool, pick confident anchor patches
from the host's own outputs, aggregate frozen DINO features into per-class
POSITIVE visual prototypes, fuse via cosine lookup. Verdict: mechanism
orthogonal (positive visual prototypes vs our background negative-word
selection + derived budget; no negatives, no budget concept) -> fourth
component remains SAFE, BUT the "unlabeled deployment pool x host output
statistics" pipeline skeleton is now taken -> narrative narrowed in
sec:winstat to "first to use it for negative selection / budget derivation";
TPA added as concurrent-work citation (tpa2026) alongside FLOSS as the two
hard comparisons. TPA has no code yet -> not eligible for comparison tables
per anchor protocol (related work only).
Other fronts: PTC still v2, OVEarth-Bench still v2, ActiveSAM/EOVSAM
unchanged; per-patch/dynamic negative gating blank for a 5th consecutive
round; "when to expand negatives" unlabeled criterion still unoccupied
(TPA's confident-anchor picking is another test-time-evidence precedent,
consistent with TTABC). Code status carried over: SCI-CLIP official code
released (eligible candidate pending exact reproduction); MAVISEG no code.
Paper action: REVA sec:winstat prior-art parenthetical updated + tpa2026
bibitem added; zero-warning recompile.

## SCI-CLIP anchor reproduction — setup + first cell (2026-08-12)
Protocol: unmodified official repo (github.com/mzamini92/SCICLIP, single
08-05 commit), path-only config edits (data_root, dino_weights_path=None
to use torch.hub dino_vitb8 cache), env `prism` (py3.10, torch 2.7.0
cu118, mmcv 2.2.0 source-built with conda cuda-nvcc 11.8, mmseg 1.2.2
with mmcv upper-bound patched 2.2.0->2.3.0 — env-level shim, recorded as
environment diff). sam2_hiera_large.pt md5 08083462423be3260cd6a5eef94dc01c.
- VOC-21: bank built from train.txt (1464 imgs, 1517 segments kept),
  full val eval (1449 imgs): **mIoU 75.91 vs published 75.9 -> |d|=0.01,
  PASSES the |d|<=0.5 anchor gate.**
  (runs/prism_voc21, logs prism_voc21_{bank,eval}.log)
- ADE-150: bank build + eval pipeline launched (train split present),
  log prism_ade_pipeline.log — pending.
- Context-60 blocker: local VOC2010 has only VAL Context masks (5105);
  bank needs TRAIN Context annotations — must convert from
  pc_trainval_mat.tar.gz (trainval .mat labels) before the bank build.
- COCO-Object blocker: only val2017 images local; bank needs train2017
  images (~19 GB download) + train semantic maps.
No vocabulary changes made; this is the published-value reproduction
gate only (precondition for comparison-table entry).

### SCI-CLIP anchor cells 2+ (2026-08-12, cont.)
- ADE-150: bank from training split (20210 imgs, 126536 segments), full
  val eval (2000 imgs): **mIoU 29.83 vs published 29.8 -> |d|=0.03 PASS.**
  (runs/prism_ade)
- Context-60 unblocked: train Context masks generated from
  pc_trainval .mat via the audit's own 459->60 LUT (convention verified:
  5/5 val masks byte-identical to distributed SegmentationClassContext
  pngs); 4998 train masks written + train.txt. Bank+eval queued
  (prism_chain.sh).
- COCO-Object unblocked: train2017 images (19.3 GB) fetched on the dev
  box and rsynced to the data disk; coco_stuff164k layout assembled via
  symlinks; GroupViT cvt_coco_object.py conversion run for all 123287
  masks (script needed a mmcv2/mmengine progress-API shim — recorded as
  env diff, conversion logic untouched). Bank+eval queued after ctx60.
Gate so far: VOC 75.91/75.9 PASS, ADE 29.83/29.8 PASS; ctx60 + cocoobj pending.

## 08-12 periodic scan (intel session, Wednesday batch, archived stage11_phase3/intel_scan_20260812.md)
All fronts CLEAR: TPA still v1/no code (related-work only), SCI-CLIP repo
HEAD unchanged at ab88844 (our reproduction commit anchor holds), PTC v2,
OVEarth-Bench v2, ActiveSAM/EOVSAM unchanged, negative gating blank for a
6th round, "when to expand negatives" unlabeled criterion still unoccupied.
Only notable: GeoSeg-OV (2608.10426, remote-sensing training-based, no
collision). Next scan window: 08-13/08-14 batch.

### SCI-CLIP anchor cell 3: Context-60 — GATE FAIL (2026-08-12)
Bank from converted train Context masks (4998 imgs, explicit
--reference-ann-file train.txt per README), full val eval (5105 imgs):
mIoU **47.66 vs published 46.1 -> |d|=1.56 > 0.5 tolerance -> FAIL**
(direction: we reproduce HIGHER than published). Frozen rule applies:
one cell outside tolerance -> SCI-CLIP does NOT enter formal comparison
tables; recorded as deviation. Candidate causes (not investigated, no
retuning allowed): reference-split ambiguity (authors' Context train
preparation may differ; our masks converted from pc_trainval .mat,
convention byte-verified on val), bank stochasticity, env diff
(torch 2.7/cu118 vs unknown). COCO-Object cell still queued for the
record; regardless of its outcome the four-cell gate cannot pass.

Context note (post-hoc observation, verdict unchanged): the +1.56
Context-60 offset matches the known systematic pattern of our converted
Context ground truth — official Trident/RF-CLIP/CorrCLIP arms also
landed +1.5~+1.6 above published on this dataset (audit paper discloses
this and draws no Context conclusions for those methods). Under the
audit paper's own frozen 1.5 anchor gate this cell would still be
marginally outside (1.56). Verdict stays FAIL per the SCI-CLIP plan's
frozen 0.5 gate; any future inclusion requires a fresh preregistration
(e.g. per-dataset exclusion mirroring the existing anchors' protocol),
not a retroactive gate change.

### SCI-CLIP anchor cell 4 + final gate verdict (2026-08-13)
- COCO-Object: bank from train2017 (118287 imgs, 290943 segments;
  GroupViT-converted _instanceTrainIds maps), full val2017 eval
  (5000 imgs): **mIoU 45.20 vs published 45.2 -> |d|=0.00 PASS.**
  (runs/prism_cocoobj, log prism_cocoobj_pipeline.log)
- FINAL four-cell gate: VOC 75.91/75.9 PASS (0.01), ADE 29.83/29.8 PASS
  (0.03), COCO-Object 45.20/45.2 PASS (0.00), Context-60 47.66/46.1
  FAIL (1.56 > 0.5). **Gate NOT passed -> SCI-CLIP excluded from formal
  comparison tables per frozen rule.** The three passing cells are
  exact; the failing cell matches the known +1.5~+1.6 systematic offset
  of our converted Context GT (same as official Trident/RF-CLIP/CorrCLIP
  arms). Any future inclusion requires a fresh preregistration mirroring
  the audit anchors' per-dataset-exclusion protocol; no retroactive gate
  change.

## 08-13 periodic scan (Thursday batch, archived stage11_phase3/intel_scan_20260813.md)
No collision: negative gating blank 7th round; unlabeled negative
selection/derived budget unoccupied; "when to expand negatives"
criterion still open. TPA v1/no code; SCI-CLIP HEAD ab88844 unchanged;
PTC/OVEarth v2; ActiveSAM/EOVSAM unchanged. Watch-only: Zero-OVCD
(2608.11663), OVIS pseudo-label work (2608.11681).

### SCI-CLIP v2 registration executed: 7th official-code anchor (2026-08-13)
Per prereg_sciclip_anchor_v2.md (fresh registration mirroring the
audit's frozen 1.5 anchor gate + per-dataset exclusion; v1 0.5-gate
FAIL verdict preserved unchanged): SCI-CLIP enters the anchor section
on VOC/COCO-Object/ADE (Context-60 excluded, +1.56 disclosed with the
same no-conclusions sentence as Trident/RF-CLIP/CorrCLIP).
Perturbation arms (official repo, name_path swap only, frozen banks
reused; runs/sciclip_*):
- VOC-21:      official 75.91 | plain 57.54 (NEG +18.4) | syn100 52.23 (drop 5.3)
- COCO-Object: official 45.20 | plain 41.75 (NEG +3.5)  | syn100 32.37 (drop 9.4)
- ADE-150:     official 29.83 | plain 29.49 (NEG +0.3)  | syn100 15.29 (drop 14.2)
Findings: retrieval family shows the same NEG/synonym dissociation
ordering as Trident (NEG down, synonym drop up across datasets) and
the largest author-code synonym drop we measure (14.2, ADE). Residual
bank coupling (label_features from official names) recorded in prereg
and disclosed in-paper. Written into sec:anchor as seventh anchor +
new bib entry zamini2026sciclip; zero-warning recompile.

### R2 review of SCI-CLIP 7th-anchor batch (2026-08-13): verdict upheld
All five checks passed (v1 FAIL preserved / v2 legitimate fresh
registration, not shopping; six cells + deltas recomputed bit-identical;
coupling disclosure code-verified; ctx exemption template consistent).
Two mandatory edits applied to sec:anchor: M1 coupling caveat expanded —
deltas on this anchor read as upper bounds (coupling may also inflate
perturbed-arm penalties, incl. headline 14.2); M2 family-level claims
qualified as anchored to this single official codebase (n=1).
Zero-warning recompile. Review archived: r2_review_sciclip_anchor.md.

## 08-14 periodic scan (Friday batch, archived stage11_phase3/intel_scan_20260814.md)
No collision, nothing red-flagged: negative gating blank 8th round;
unlabeled negative selection/derived budget unoccupied; "when to
expand negatives" criterion still open. TPA v1/no code; SCI-CLIP
unchanged (paper v1, repo HEAD ab88844 — 7th-anchor baseline intact);
PTC/OVEarth v2; ActiveSAM/EOVSAM unchanged.

## 2026-07-30 投稿前终审 MUST-FIX 落实（三子会话报告回收）
Audit paper (stage5_paper/arxiv_v1/main.tex):
- Removed dangling claim IDs C1/C5, C3, C6, C7/C8 (rewritten in words).
- Distractor range 6--10 -> 5--10.
- Added global "deltas computed from unrounded run records (may differ from
  displayed rounded values by <=0.1)" statement in protocol section + NEG
  column note in tab:robustbench caption. Verified against RESULTS raw
  values that all flagged 0.1-level deltas (RF-CLIP +23.8/4.1, ProxyCLIP
  +14.0, Trident 5.8, NACLIP +5.9, ViT-L 5.7, NEG 22.1/19.6) are the
  correct unrounded-record values.
- Trident Context-60 inclusive-gate wording confirmed in place.
- SCI-CLIP mechanism description corrected (region-pooled segment
  embeddings; training-split bank + SAM2 flagged as code-level description
  of the shipped official configuration).
- Bib: zamini2026sciclip title+second author (Zamini, Shukla; Segment-Centric
  Inference with Reference Memory...); beyondbench2026 real authors (Jia,
  MaungMaung, Nguyen, Chen, Echizen; ICPR 2026); added stojnic2025lposs and
  bai2024scclip entries + in-text \citep at the LPOSS/SC-CLIP official-anchor
  sentences; zhou2023rethinking label year 2023->2025.
- Remaining before submission: author list/affiliations placeholder (needs
  Thomas), human-vocabulary collection (15 real participants, external).
REVA paper (stage8_method_paper/main.tex):
- M1: +4.7..+5.9 relabelled as under region arbitration (Table 1, REVA vs
  rand.64+SAM).
- M2: ADE endpoints given at two decimals (13.13->14.17, 14.74->16.06);
  +1.0/+1.3 confirmed correct from unrounded records.
- M3: abstract safety criterion spelled out (no class drops >3 IoU on two or
  more methods; worst single-method regression -3.2).
- M4: off-VOC hand-list endpoints at two decimals (30.48/30.65 vs
  31.81/32.81); +1.3/+2.2 confirmed from unrounded records.
- M5: +3.5 arbitration gain annotated (59.62->63.16), correct unrounded.
- M6: tab:perclass caption fixed (tvmonitor all four methods, person three
  of four, MaskCLIP +8.6).
- M7: three wrong \ref targets fixed (sec:controls -> Table tab:main x2;
  sec:prereg -> sec:method for the per-patch win-statistics diagnostic).
- Added the same global unrounded-records delta statement in sec:prereg.
Both papers recompiled: no undefined references/citations; only benign
float-specifier warnings. SHOULD-FIX items (spelling unification, abstract
length, S-level metadata) deferred, recorded in /tmp reports.

## W39 full-split replication of subset-based key cells (prereg_w39_fullsplit_repl.md) — ALL FOUR GATES PASS
Runs: runs/w39_ade_{sclip,naclip}_{plain,sam}.json,
runs/w39_vitl_{sclip,naclip}_{plain,official,sam}.json (2026-08-25).
Part A, ADE-150 full val (2000 imgs, ViT-B): SCLIP plain 16.66,
pix_vabs 16.31, sam_reg_vabs 17.83, sam_reg_rand 17.93; NACLIP plain
17.61, pix_vabs 17.26, sam_reg_vabs 19.14, sam_reg_rand 19.12.
- H-A1 (pooling >= +0.5 both hosts): +1.53 / +1.88 -> PASS. Paper
  "+1.0/+1.3, val first-300" sentence upgraded to full-val numbers.
- H-A2 (VABS null boundary, both |d| <= 2.0): pix_vabs - plain
  -0.35 / -0.35; sam_vabs - sam_rand -0.10 / +0.03 -> PASS. Boundary
  claim confirmed at full val; paper now cites full-val cells.
Part B, VOC-21 dev-excluded full split (1349 imgs, ViT-L-14-quickgelu):
SCLIP plain 37.72, official 41.99, pix_vabs 42.18, REVA 45.87,
rand64+SAM 43.40; NACLIP plain 35.71, official 51.59, pix_vabs 48.04,
REVA 53.14, rand64+SAM 49.10.
- H-B1 (REVA - plain >= +3.0 both hosts): +8.14 / +17.44 -> PASS.
- H-B2 (REVA >= official - 0.5 both hosts): 45.87 vs 41.49 and
  53.14 vs 51.09 -> PASS. Above-official-pixel sentence upgraded to
  full split.
Descriptive only (frozen as no-claim): VABS-vs-rand under arbitration
at full split is +2.46 (SCLIP) / +4.05 (NACLIP); the earlier test-300
SCLIP value (+1.8, below the +2 bar) is retained as disclosure; the
selection-advantage claim scoping is NOT changed by W39 (single seed,
ViT-B-selected negatives reused).
Provenance note (adversarial review): the four w39_*_sam.json records
carry a stale "prereg" field ("prereg_v5.md", the probe's original
prereg) and do not self-record the model name / vocab paths / seed;
those are fixed by this entry and by queue_w39.sh (archived on the
temp-hb mirror): Part A = ViT-B-16-quickgelu with
ade150_plain_vabs64.json / ade150_plain_randneg64.json, Part B =
ViT-L-14-quickgelu with voc21_plain_vabs64.json /
voc21_plain_randneg64.json, archived seed-0 random set throughout.
All deltas above recomputed from unrounded records.
