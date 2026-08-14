# Prereg W28: crowding signal on an independent backbone (ViT-L/14) --- frozen before runs

Date frozen: 2026-08-04, after the W27 R2 review.
R2's residual objection to W27-H2: the three hosts share the ViT-B/16
CLIP backbone and produce IDENTICAL rank orderings of both S1 and gain,
so the per-host stratified pass is one test replicated three times. W28
breaks the shared-backbone degeneracy with the only independent backbone
already validated in-stack (W1a): ViT-L-14-quickgelu / openai.

## Protocol (frozen; identical to W27 except the backbone)
- Hosts: sclip and naclip on ViT-L-14-quickgelu (openai weights);
  clearclip excluded (W1a validated only sclip/naclip attention edits
  on ViT-L).
- Same 6 cells per host as W27: {voc21, cocoobj, ctx60, ade150}/plain +
  {voc21, cocoobj}/syn100_s0 -> 12 points.
- S1: mean max raw cosine (50 unlabeled images, GT never read).
- Oracle gain: test-300, single seed; VABS-64 negatives RE-SELECTED
  with the ViT-L text encoder for each vocabulary (the selector's
  embedding space must match the host); new vocab files suffixed
  _vabs64_vitl.json.
## Criteria (frozen)
- H1: per-host Spearman <= -0.5 for BOTH hosts, with at least one host
  whose (S1, gain) rank ordering differs from the W27 ViT-B ordering
  in S1 (i.e. the test is not a third copy of the same ranking) ->
  INDEPENDENT REPLICATION PASS. If orderings are again identical to
  ViT-B's AND rho passes, report PASS-BUT-DEGENERATE (no independence
  claim). Wrong-signed rho on any host -> FALSIFIED.
- H2 (vocabulary axis, descriptive only after W27's 0/6): report the
  4 plain-vs-syn100 pairs; no pass bar --- W27 already falsified the
  vocabulary-level claim and W28 cannot un-falsify it (a reversal here
  would be reported as a backbone-dependence observation, not a pass).
## Caveats frozen
Single seed, test-300, one synonym draw, two hosts only; ViT-L shares
the training data/objective with ViT-B (same openai CLIP release), so
"independent backbone" means independent weights/width/patch size, not
an independent pretraining distribution.
