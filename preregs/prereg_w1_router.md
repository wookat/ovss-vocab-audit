# Pre-registration W1b: few-label per-region config router (frozen 2026-07-31)

Context: the C6 oracle showed a +14.9 mIoU per-region routing ceiling on dev-100,
but every label-free gate failed (all below best single config). Hypothesis: a
SMALL amount of supervision (50 labelled images) suffices to learn a region router
that recovers a useful fraction of the ceiling. This deliberately leaves the
training-free regime and is disclosed as a few-label method.

Setup (probe_route_ws.py):
- Protocol: unified dense, VOC-21 official vocab, single-crop 336 (as C6 oracle).
- Pool: top-4 configs of the 20 by mIoU on the TRAIN images only (dev 1-50).
- Regions: SAM ViT-B pps=16, pixel fallback = router-chosen best-margin config.
- Features per (region, config), all computable at inference: region mean
  top1-top2 margin, region mean top1 prob, region mean entropy, log region size,
  agreement fraction of this config's region label with the pool majority label,
  config one-hot.
- Router: ridge regression predicting region pixel accuracy, fit on dev 1-50
  regions; route each region to argmax predicted accuracy.
- Eval: dev 51-100 (held-out) AND test-300 (offset 0-300, fully disjoint).

Kill / go criteria (frozen):
- K-W1b-a (go): routed mIoU on test-300 >= best single pool config on test-300
  + 2.0 mIoU.
- K-W1b-b: routed must beat every pool member on held-out dev too; report the
  fraction of the (train-measured) oracle ceiling recovered.
- If gain in [0, 2): one disclosed redesign (feature set or router class) allowed.
- Negative-transfer guard: if routed < best single on either eval set, killed.
No claim of training-free; supervision cost (50 images) disclosed. A pass licenses
scaling the router (more features/configs, other datasets) under a new prereg.
