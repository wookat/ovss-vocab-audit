# Preregistration W4c: Why does label propagation smooth naming noise? (frozen 2026-08-01)

Candidate B from Ideator-5. Two claims to falsify:

## H1 (mechanism): the robustness advantage of LPOSS-style propagation under
vocabulary perturbation is explained by low-pass filtering of text-side noise
over the DINO affinity graph: perturbed-vocab logit maps differ from
plain-vocab logit maps mainly in high-frequency (graph-spectral) components,
which propagation attenuates.

Test E1: on VOC test-300, for ClearCLIP base logits under plain vs syn100
vocab, decompose the logit difference into graph-frequency bands (eigenvectors
of the kNN DINO graph Laplacian, k=32, per image, 20 images sufficient for
spectra). PASS if >=60% of the perturbation-induced logit-difference energy
lies in the top (high-frequency) half of the spectrum, and propagation
(alpha=0.9, 10 iters) attenuates that band by >=2x more than the low band.

## H2 (transplant): the smoothing benefit is separable and transplantable —
applying the same propagation operator post-hoc to the OTHER six methods'
logits recovers a consistent share of their synonym drop.

Test E2: for the 6 non-LPOSS methods on VOC test-300, run {plain, syn100}
with and without post-hoc propagation. GO if propagation reduces the mean
synonym drop (plain - syn100) by >=25% averaged over methods without reducing
plain mIoU by more than 1.0 on any method.

## Kill criteria
- KILL H1 if energy is not concentrated high-frequency (<50%) or attenuation
  ratio < 1.5x.
- KILL H2 if mean drop reduction < 15% or any method loses > 2.0 plain mIoU.
- If H1 passes and H2 fails (or vice versa) report as partial mechanism
  finding for the audit paper appendix, no standalone paper.

## Disclosures
LPOSS-style is our style-reimplementation; all claims are within-protocol.
No changes to frozen robust-mIoU definitions.
