# Prereg W25: NEG-Thermostat H1 --- unlabeled image/prediction-side signals vs oracle negative-expansion gain (frozen before runs)

Date frozen: 2026-08-04, after W24 closed (H1 falsified, H2 marginal) and
before any W25 computation. Fallback declared in W24 verdict: image-side
signals are an independent source from the (marginal) text-side angle.

## Question
Can a deployment-time, label-free signal computed from the host's own
predictions on a handful of images predict whether vocabulary-adaptive
negative expansion (VABS-64) will help or harm? If yes, this closes the
REVA limitation "applicability (NEG headroom) cannot be measured at
deployment" (intel scan 2026-08-03: direction SAFE, no prior art for
dosage/applicability self-detection in OVSS negative mechanisms).

## Protocol (frozen)
- Hosts: SCLIP, NACLIP (in-stack pixel protocol, ViT-B/16).
- Configs (8 points, matched test-300 pixel protocol, single seed), oracle
  gain = archived VABS64 minus plain mIoU:
  sclip/voc21 +18.87, naclip/voc21 +16.15 (runs/w24_snb.json);
  sclip/cocoobj +9.24, naclip/cocoobj +10.73 (W14-N3);
  sclip/ctx60 +0.5, naclip/ctx60 +0.6 (W13-L5 transfer block);
  sclip/ade150 -0.7, naclip/ade150 -1.0 (same block).
- Signals: computed on the FIRST 50 images of each dataset's eval list
  (unlabeled; GT never read), plain vocabulary, standard windowed dense
  inference:
  S1 crowding: mean over pixels of the max cosine similarity to the
     vocabulary queries. Pre-declared sign: NEGATIVE (crowded vocabulary
     = no unmodeled background = no headroom).
  S2 uncertainty: mean per-pixel softmax entropy (scale 40, plain
     queries). Pre-declared sign: POSITIVE.
  S3 background energy: mean squared projection norm of patch features
     onto the W24 background basis (r=32, frozen lexicon). Pre-declared
     sign: POSITIVE.
- Criterion (frozen): PASS if at least one signal reaches |Spearman| >=
  0.6 across the 8 points WITH its pre-declared sign; FALSIFIED if all
  three have |rho| < 0.3 or only wrong-signed correlations >= 0.3.
- If PASS: a follow-up H2 (frozen now): threshold the passing signal at
  the midpoint between the harmful cluster and the helpful cluster
  (leave-one-out), auto-switch k in {0, 64}; do-no-harm requires loss
  <= 0.2 mIoU on harmful configs and >= 90% of oracle gain retained on
  helpful configs, evaluated on the same 8 points (acknowledged: small n,
  same-point selection+evaluation; would be reported as a pilot, with the
  official-code anchors (Trident/CorrCLIP COCO, Trident Context) reserved
  as the honest out-of-sample check in a later wave).
## Caveats frozen in advance
8 points, 4 datasets, 2 hosts sharing datasets (signals may be nearly
host-independent -> effective n closer to 4); the prior text-side
per-class predictor was NULL and the W24 angle was marginal; failure is a
live outcome and will be preserved either way.
