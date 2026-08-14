# Phase-3 oracle probes (prereg_c2c6_oracle.md), VOC dev-100, official vocab

## C2 HeadText subspace-decomposed similarity — KILLED
c2_headtext_oracle.json (fit w on dev 1-50, eval on dev 51-100):
- vanilla: oracle 34.71 vs all-heads baseline 33.61 -> gain +1.11 (< +3.0, K-C2a FAIL;
  also below E1 vanilla oracle head-selection +2.8, K-C2b FAIL).
- qq: oracle 47.91 vs baseline 53.30 -> gain -5.38 (FAIL).
- Uniform-w decomposition (sum of projected sims) loses 2-6 mIoU vs <t,v>:
  head subspaces overlap; the decomposition itself destroys cross-subspace
  interference terms that carry signal. Direction dead per prereg.

## C6 per-region config routing — ORACLE PASS
c6_route_oracle.json (20 configs = 4 flavors x L8-12; SAM ViT-B regions pps=16):
- best single config: kk_L12 = 53.23
- per-image oracle: 57.53 (+4.31)
- per-region oracle: 68.17 (+14.94 >= +4.0, K-C6a PASS)
- Region granularity adds +10.6 beyond image-level (K-C6b: granularity matters).
Caveats (as prereg): oracle uses GT; no method claim. Next step requires a separate
pre-registration for a label-free per-region gate (margin predictor is validated only
at whole-config level, Spearman 0.89; per-region is untested). The 20-config expert
pool costs ~20x dense passes; a distilled/pruned pool is needed for practicality.

# Round 2 (prereg_c6r2_c4.md), VOC test/dev as specified

## C6 round-2 label-free routing — KILLED
c6r2_route_lf.json (pool = label-free top-4 by E2 margin: qq/ident/vanilla/kk @L12;
SAM regions; dev-100 official vocab):
- pool members: kk_L12 53.23, qq_L12 51.76, ident_L12 42.81, vanilla_L12 30.34
- gates: G1 margin 40.78, G2 confidence 38.21, G3 agreement 46.68
- 4-pool oracle (with label-free fallback on uncovered pixels): 50.05
- ALL arms below best single config (53.23) -> < +2 zone -> direction killed per
  prereg. Two mechanisms: (i) the label-free margin misroutes at region granularity
  (consistent with E2 K-E2b: the margin signal is global, not localized); (ii) the
  label-free pool selection itself admits vanilla_L12 (30.3) which poisons routing.
  The +14.9 oracle ceiling exists but is not reachable with our label-free signals.

## C4 vocabulary-perturbation consistency (visual side) — KILLED
Frozen NACLIP baselines (test-300): plain 36.54, official 55.02, held-out syn100
29.67. Training run 1 (prereg hypers) aborted by drift monitor after ep1
(drift 0.566 > 0.5). Disclosed redesign (lr 5e-5, w_anchor 2.0, w_drift 1.0)
aborted at ep2; its ep-1 checkpoint (drift 0.466, within monitor) evaluated:
- plain 30.21 (-6.3, K-C4a FAIL: needed >= +2)
- official 54.13 (-0.9, K-C4b borderline pass)
- held-out synonym 25.30 (-4.4: HARM, same signature as the text-side failure)

## Consolidated phase-3 conclusion
The naming-invariance/consistency objective now fails in BOTH embedding spaces:
text-side adapters (LexRO) and visual-side adapters (C4) each damage held-out
synonym vocabularies while degrading plain performance. Combined with whitening
(linear), learned adapters (text), learned background queries, and now visual
consistency training, the evidence supports the strong claim: naming robustness is
not learnable in isolation on either side of the CLIP embedding space with
unlabeled data; the only surviving repairs remain inference-time and
vocabulary-adaptive (VABS) or region-evidence-based (SAM arbitration).
Head-subspace decomposition (C2) and label-free routing (C6) add two more
mechanism-level negative results with informative oracle gaps.
