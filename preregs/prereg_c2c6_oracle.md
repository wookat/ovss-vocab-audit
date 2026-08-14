# Pre-registration: Phase-3 oracle upper-bound probes (C2 HeadText, C6 config routing)

Frozen before any experiment. Date: 2026-07-30. Protocol: unified dense protocol
(OpenCLIP ViT-B/16 openai, short=336, single-crop 336 for probes as in HeadLens E1/E2),
VOC-21 dev-100 (offset 300), official vocabulary, logit scale 40.

## C2 HeadText oracle (probe_headtext_oracle.py)

Hypothesis: the dense-quality signal lives in per-head *subspace alignment*
sim(t,v) = sum_h w_h <P_h t, P_h v>, not in whole-head selection. P_h = orthogonal
projector onto the column space of B_h = proj^T diag(ln_post.weight) W_O[:, h-block]
(linearised ln_post; disclosed approximation). Patch embedding v = full all-heads
embedding (exact pipeline output); text t = template-averaged class embedding.

Oracle: fit global weights w (12 params, softmax cross-entropy on patch-resolution GT)
on dev images 1-50, evaluate mIoU on dev images 51-100. Flavors: vanilla and qq
(surgery representative), output-only last block at L12 (same machinery as E1).

Kill / go criteria (frozen):
- K-C2a (go): oracle-weighted decomposed similarity beats the all-heads baseline
  <t,v> by >= +3.0 mIoU on the held 50 images for at least one flavor.
- K-C2b (context): the gain must exceed the E1 oracle head-selection gain on the same
  flavor (+0.4~+0.6 surgery / +2.8 vanilla); otherwise the decomposition adds nothing
  beyond selection and the direction is killed.
- Diagnostics recorded regardless: learned w spectrum, per-head projected-similarity
  quality, uniform-w (=1) sanity arm (should approx reproduce baseline).

## C6 per-region config routing oracle (probe_config_route_oracle.py)

Hypothesis: different (flavor x exit-layer) configs win on different regions; a
label-free margin gate could route per region. First establish the oracle ceiling.

Configs: the 20 E2 configs (4 flavors x exit layers 8..12). Regions: SAM ViT-B
automatic masks (points_per_side 16), uncovered pixels = own singleton region set
via pixel fallback of the routed config chosen per image.
Oracle routing: per region choose the config maximizing region pixel accuracy vs GT;
also record per-image oracle (choose best config per image).

Kill / go criteria (frozen):
- K-C6a (go): per-region oracle mIoU > best single config + 4.0 mIoU on dev-100.
- K-C6b (context): per-image oracle reported; if per-region ~= per-image, routing
  granularity adds nothing.

Both probes are oracle upper bounds (GT used); no method claim follows from a pass —
a pass only licenses building the label-free gate / weight predictor next, with a
separate pre-registration. A fail kills the direction.
