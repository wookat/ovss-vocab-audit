"""W4d: presence-gated REVA (prereg_w4d_presence.md, frozen).

Per vocab item c: presence score s(c) = mean of top-K (K=3) SAM-region pooled
probabilities. Gate by within-vocabulary relative rank: keep classes with
s(c) >= tau * median(s); pixels of rejected classes re-argmax over survivors.
Arms per image: ungated pixel, ungated SAM-region, gated SAM-region.
Also logs gate precision/recall against GT presence (evaluation only).
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from probe_d1sam import build_region_map, pool_regions


def region_class_scores(probs, reg):
    """Mean region-pooled class probabilities; returns (R,K) tensor or None."""
    K = probs.shape[0]
    if reg.max() < 0:
        return None
    flat = probs.reshape(K, -1)
    r = torch.from_numpy(reg.reshape(-1)).to(probs.device)
    valid = r >= 0
    R = int(r.max()) + 1
    out = torch.zeros(R, K, device=probs.device)
    cnt = torch.zeros(R, device=probs.device)
    out.index_add_(0, r[valid], flat[:, valid].T)
    cnt.index_add_(0, r[valid], torch.ones(int(valid.sum()), device=probs.device))
    return out / cnt.clamp_min(1).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="clearclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--n-gt-classes", type=int, required=True,
                    help="first N vocab entries are the GT class space")
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points_per_side)

    if a.variant == "proxyclip":
        from proxyclip_seg import ProxyCLIP
        model = ProxyCLIP()
    elif a.variant == "lposs":
        from newgen_seg import LPOSS
        model = LPOSS()
    elif a.variant == "scclip":
        from newgen_seg import SCCLIP
        model = SCCLIP()
    else:
        model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]

    names = json.load(open(a.vocab_file))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    Kv = int(qi.max()) + 1
    Kg = a.n_gt_classes
    meters = {k: IoUMeter(Kv, ignore) for k in ["pix", "sam_reg", "sam_gate"]}
    scale = 40.0
    tp = fp = fn = 0

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        probs_q = (scale * sims).softmax(0)
        probs = torch.zeros(Kv, *sims.shape[1:], device=sims.device)
        idx = qi.to(sims.device).view(-1, 1, 1).expand_as(probs_q)
        probs.scatter_reduce_(0, idx, probs_q, reduce="amax", include_self=False)

        meters["pix"].update(probs.argmax(0).cpu().numpy(), gt)
        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        meters["sam_reg"].update(pool_regions(probs, reg), gt)

        rc = region_class_scores(probs, reg)
        if rc is None:
            meters["sam_gate"].update(pool_regions(probs, reg), gt)
            continue
        k_top = min(a.topk, rc.shape[0])
        s = rc.topk(k_top, dim=0).values.mean(0)  # (Kv,) presence score
        keep = s >= a.tau * s.median()
        keep[0] = True  # never gate background/class 0
        pres_gt = torch.zeros(Kv, dtype=torch.bool)
        gtt = torch.from_numpy(np.asarray(gt))
        for c in range(Kg):
            pres_gt[c] = bool((gtt == c).any())
        pred_pres = keep.cpu()[:Kg]
        tp += int((pred_pres & pres_gt[:Kg]).sum())
        fp += int((pred_pres & ~pres_gt[:Kg]).sum())
        fn += int((~pred_pres & pres_gt[:Kg]).sum())

        gated = probs.clone()
        gated[~keep] = 0.0
        meters["sam_gate"].update(pool_regions(gated, reg), gt)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}] pix={meters['pix'].miou()[0]*100:.2f} "
                  f"reg={meters['sam_reg'].miou()[0]*100:.2f} "
                  f"gate={meters['sam_gate'].miou()[0]*100:.2f}", flush=True)

    res = {"prereg": "prereg_w4d_presence.md", "variant": a.variant,
           "dataset": a.dataset, "vocab_file": a.vocab_file,
           "tau": a.tau, "topk": a.topk, "n": len(samples),
           "gate_precision": tp / max(tp + fp, 1),
           "gate_recall": tp / max(tp + fn, 1), "arms": {}}
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "miou_all": m.miou_all()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({**{k: round(v["miou_all"] * 100, 2) for k, v in res["arms"].items()},
                      "P": round(res["gate_precision"], 3),
                      "R": round(res["gate_recall"], 3)}, indent=1))


if __name__ == "__main__":
    main()
