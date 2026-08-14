# Prereg W27: crowding signal --- independent-point expansion under a matched extraction protocol (frozen before runs)

Date frozen: 2026-08-04, after the R2 incremental review of W24-W26.
R2's residual objections to the applicability-signal line: (a) effective
n ~ 4 (hosts share datasets, all points are plain-vocabulary), (b) W26
extraction-depth heterogeneity. W27 addresses both with NEW points that
vary the vocabulary (not just the dataset) and a single matched
extraction protocol for every point.

## Protocol (frozen)
- Extraction, identical for all points: in-stack pixel protocol,
  ViT-B/16; S1 crowding = mean over pixels (first 50 unlabeled images,
  GT never read) of the max RAW COSINE similarity between the dense
  patch embedding and the vocabulary query embeddings. No softmax
  anywhere in the signal path.
- Oracle gain, identical for all points: test-300, standard meter,
  single seed; gain = mIoU(vocab + VABS-64 conditioned on that vocab)
  minus mIoU(vocab alone). VABS negatives are re-selected per vocabulary
  with the frozen vabs.py selector (M=64 via background sub-query
  mechanics identical to all previous in-stack VABS runs).
- Points (18): hosts {sclip, naclip, clearclip} x
  {voc21/plain, cocoobj/plain, ctx60/plain, ade150/plain,
   voc21/syn100_s0, cocoobj/syn100_s0}.
  ALL 18 cells are recomputed fresh under this matched protocol
  (including the sclip/naclip plain cells) so no point mixes protocols;
  archived values are used only as sanity cross-checks.
- The syn100 cells are the genuinely new axis: same dataset and host,
  different vocabulary -> tests whether the signal tracks the
  vocabulary (as claimed) rather than dataset identity.
## Criteria (frozen)
- H1 (pooled): Spearman(S1, gain) <= -0.6 over the 18 points -> PASS;
  > -0.3 -> FALSIFIED; between -> MARGINAL.
- H2 (stratified, only if H1 not falsified): per-host Spearman <= -0.5
  for ALL three hosts -> STRATIFIED PASS (this is what W25 could not
  show); any host with wrong-signed rho -> stratification FAIL,
  reported.
- H3 (vocabulary axis): within each (host, dataset) plain-vs-syn100
  pair (6 pairs), the vocabulary with higher S1 has lower gain ->
  count of correct pairs; >= 5/6 PASS, <= 3/6 FAIL, 4/6 MARGINAL.
## Caveats frozen
Single seed, test-300, 50-image signals; syn100_s0 is one synonym draw;
clearclip shares the CLIP backbone with sclip/naclip (host diversity is
architectural at the attention level only); failure preserved either
way. Falsification here kills the "signal tracks vocabulary headroom"
claim and the pilot stays a dataset-level observation.
