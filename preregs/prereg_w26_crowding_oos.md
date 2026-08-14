# Prereg W26: crowding signal out-of-sample --- within-host ordinal test on official codebases (frozen before runs)

Date frozen: 2026-08-04, after W25 closed (crowding S1 passed its
correlation criterion in-stack; global-threshold switch failed do-no-harm
because signal scales are host-dependent).

## Question
Does the label-free crowding signal, computed from each OFFICIAL
pipeline's own dense predictions, order that host's datasets by oracle
VABS-64 gain? Ordinal-within-host is deliberately scale-free: W25 showed
absolute thresholds do not transfer across hosts, so no threshold or
switch is tested here; only the ordering claim.

## Protocol (frozen)
- Hosts / configs (7 points, oracle gains archived from W16-C/D, W17/B/C,
  W21; official code, plain name files, full-val gains):
  scclip: voc21 +16.65, cocoobj +3.54
  trident: voc21 +16.04, cocoobj +1.28, ctx60 -3.41
  corrclip: voc21 +15.24, cocoobj -4.08
- Signal S1 (crowding): mean over pixels and over the FIRST 50 images of
  the dataset's official val list of the maximum per-pixel class score,
  taken from the pipeline's stored per-class map (`seg_logits` /
  `data_samples.seg_logit`) for the PLAIN name file, before any
  background thresholding, forward completely unmodified (read-only
  consumption exactly as in W16-E / W23). Per-host maps are internally
  comparable across datasets (same code path and scale); no cross-host
  comparison is made.
- All three repos store POST-SOFTMAX probability maps, so the max-prob
  signal carries a mechanical class-count confound (fewer classes =>
  higher max prob). Disclosed in advance; note the confound works
  AGAINST the prediction on the voc-vs-coco pairs (voc has fewer classes
  AND the larger gain, so mechanics push voc's max-prob UP while the
  prediction needs voc's crowding DOWN) --- a correct ordering there is
  evidence despite, not because of, the confound. Frozen secondary
  variant to partially deconfound: margin M = mean over pixels of
  [log p_max - mean_c log p_c]. Both are computed; the PRIMARY criterion
  is evaluated on whichever of S1/M was declared here --- frozen choice:
  M primary, max-prob descriptive.
- Prediction (pre-declared sign, as W25): within each host, HIGHER
  crowding (M) => LOWER gain.
- Criterion (frozen): 5 within-host ordered pairs (scclip 1, corrclip 1,
  trident 3). PASS if all 5 pairs are ordered correctly; MARGINAL if
  exactly 1 pair is wrong (report, no claim); FALSIFIED if >= 2 pairs are
  wrong.
- Caveats frozen: n=5 pairs; single run; the stored maps differ in
  post-processing depth per repo (scclip: post softmax/alias/PAMR;
  corrclip: post softmax/background-fold) --- disclosed, per-host
  internal consistency is what matters for an ordinal test. GT never
  read; only images and the plain name files are consumed.
