"""W5a: presence-gated REVA v2 (prereg_w5a_presence2.md, frozen).

New signal family (distinct from killed W4d rank-gate):
 s1(c) = max over SAM regions of [raw pooled cosine sim(r,c)
         - max over VABS negatives of raw pooled sim(r,neg)]
 s2(c) = fraction of top-20 regions by sim(r,c) whose argmax class is c
Gate: keep c iff z(s1) >= 0 OR s2 >= 0.3 (frozen). Background always kept.
Arms: ungated pixel, ungated SAM-region, gated SAM-region.
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


def region_mean(sims, reg):
    """Mean region-pooled values; sims (K,H,W) -> (R,K) or None."""
    K = sims.shape[0]
    if reg.max() < 0:
        return None
    flat = sims.reshape(K, -1)
    r = torch.from_numpy(reg.reshape(-1)).to(sims.device)
    valid = r >= 0
    R = int(r.max()) + 1
    out = torch.zeros(R, K, device=sims.device)
    cnt = torch.zeros(R, device=sims.device)
    out.index_add_(0, r[valid], flat[:, valid].T)
    cnt.index_add_(0, r[valid], torch.ones(int(valid.sum()), device=sims.device))
    return out / cnt.clamp_min(1).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="clearclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--neg-file", required=True,
                    help="VABS meta json with a 'negatives' word list")
    ap.add_argument("--n-gt-classes", type=int, required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--points-per-side", type=int, default=16)
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
    samples, _, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]

    names = json.load(open(a.vocab_file))
    negs = json.load(open(a.neg_file))["negatives"]
    T, qi = class_embeddings(model, names + negs)
    T = T.to(model.device)
    qi = qi.to(model.device)
    Kv = len(names)
    Kg = a.n_gt_classes
    is_neg = (qi >= Kv)
    meters = {k: IoUMeter(Kv, ignore) for k in ["pix", "sam_reg", "sam_gate"]}
    scale = 40.0
    tp = fp = fn = 0

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims_q = seg_logits(model, t, T)  # (Q,H,W) raw cosine
        sims_q = F.interpolate(sims_q.unsqueeze(0), size=(h0, w0), mode="bilinear",
                               align_corners=False)[0]
        # pixel probs over vocab classes only (standard eval arm)
        sq_v = sims_q[~is_neg]
        qi_v = qi[~is_neg]
        probs_q = (scale * sq_v).softmax(0)
        probs = torch.zeros(Kv, *sims_q.shape[1:], device=sims_q.device)
        probs.scatter_reduce_(0, qi_v.view(-1, 1, 1).expand_as(probs_q), probs_q,
                              reduce="amax", include_self=False)
        meters["pix"].update(probs.argmax(0).cpu().numpy(), gt)

        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        meters["sam_reg"].update(pool_regions(probs, reg), gt)

        rq = region_mean(sims_q, reg)  # (R, Q) raw sims
        if rq is None:
            meters["sam_gate"].update(pool_regions(probs, reg), gt)
            continue
        R = rq.shape[0]
        # class-level region sims: max over sub-queries per class
        rc = torch.full((R, Kv), -1e9, device=rq.device)
        rc.scatter_reduce_(1, qi_v.view(1, -1).expand(R, -1), rq[:, ~is_neg],
                           reduce="amax", include_self=False)
        rneg = rq[:, is_neg].max(1).values  # (R,) best negative per region
        margin = rc - rneg.unsqueeze(1)  # (R, Kv)
        s1 = margin.max(0).values  # (Kv,)
        z1 = (s1 - s1.mean()) / s1.std().clamp_min(1e-6)
        # s2: winner consistency among top-20 regions per class
        k_top = min(20, R)
        winner = rc.argmax(1)  # (R,)
        top_idx = rc.topk(k_top, dim=0).indices  # (k_top, Kv)
        s2 = (winner[top_idx] == torch.arange(Kv, device=rc.device).view(1, -1)).float().mean(0)
        keep = (z1 >= 0) | (s2 >= 0.3)
        keep[0] = True

        pres_gt = torch.zeros(Kg, dtype=torch.bool)
        gtt = torch.from_numpy(np.asarray(gt))
        for c in range(Kg):
            pres_gt[c] = bool((gtt == c).any())
        pred_pres = keep.cpu()[:Kg]
        tp += int((pred_pres & pres_gt).sum())
        fp += int((pred_pres & ~pres_gt).sum())
        fn += int((~pred_pres & pres_gt).sum())

        gated = probs.clone()
        gated[~keep] = 0.0
        meters["sam_gate"].update(pool_regions(gated, reg), gt)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}] pix={meters['pix'].miou()[0]*100:.2f} "
                  f"reg={meters['sam_reg'].miou()[0]*100:.2f} "
                  f"gate={meters['sam_gate'].miou()[0]*100:.2f}", flush=True)

    res = {"prereg": "prereg_w5a_presence2.md", "variant": a.variant,
           "dataset": a.dataset, "vocab_file": a.vocab_file,
           "neg_file": a.neg_file, "n": len(samples),
           "gate_precision": tp / max(tp + fp, 1),
           "gate_recall": tp / max(tp + fn, 1), "arms": {}}
    for k, m in meters.items():
        miou, _ = m.miou()
        res["arms"][k] = {"miou": miou, "miou_all": m.miou_all()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({**{k: round(v["miou_all"] * 100, 2) for k, v in res["arms"].items()},
                      **{k + "_gt": round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                      "P": round(res["gate_precision"], 3),
                      "R": round(res["gate_recall"], 3)}, indent=1))


if __name__ == "__main__":
    main()
