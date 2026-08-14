"""W2d: SelfRoute go/no-go (stage12_month/prereg_w2d_selfroute.md).

Pool K=8 configs (4 flavors x L{11,12}). Arms per region:
  consist-argmax, consist+dino (fallback below-median crispness),
  margin-argmax (control), oracle (GT). Baseline: every pool member's mIoU.
K1: pseudo-label match rate with oracle config vs 1/8+10pp.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

import data
from probe_headlens import FLAVORS
from probe_headlens_e2 import HeadLensL
from probe_d1sam import build_region_map
from probe_route_lf import class_probs
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter


def dino_features(dino, img_r, h0, w0, device):
    t = to_tensor(img_r, device)
    # pad to multiple of 16
    _, _, H, W = t.shape
    Hp, Wp = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    t = F.pad(t, (0, Wp - W, 0, Hp - H))
    with torch.no_grad():
        feats = dino.get_intermediate_layers(t, n=1)[0][:, 1:]  # B,N,C
    gh, gw = Hp // 16, Wp // 16
    f = feats[0].T.reshape(-1, gh, gw).unsqueeze(0)
    f = F.interpolate(f, size=(h0, w0), mode="bilinear", align_corners=False)[0]
    return F.normalize(f, dim=0)  # C,h0,w0


def region_crispness(dfeat, sel_np):
    ring = binary_dilation(sel_np, iterations=3) & ~sel_np
    if ring.sum() == 0:
        return 0.0
    inside = dfeat[:, torch.from_numpy(sel_np)].mean(1)
    outside = dfeat[:, torch.from_numpy(ring)].mean(1)
    return float(1.0 - F.cosine_similarity(inside, outside, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-ckpt", default="/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                          skip_validation=True).to("cuda").eval()

    model = HeadLensL()
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    K = len(plain_names)
    samples = samples[a.offset:a.offset + a.limit]
    T, qi = class_embeddings(model, plain_names)
    T = T.to(model.device)

    pool = [(f, L) for f in FLAVORS for L in [11, 12]]
    C = len(pool)
    Hn = 12

    meters = {k: IoUMeter(K, ignore) for k in
              ["consist", "consist_dino", "margin_ctl", "oracle"]}
    single = [IoUMeter(K, ignore) for _ in pool]
    match_consist = match_margin = n_regions = 0
    crisp_all = []

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        probs, preds, gmarg = [], [], []
        for f, L in pool:
            outs, bias, (gh, gw) = model.head_outputs_at(t, f, L)
            emb = model.subset_emb(outs, bias, range(Hn))
            p = class_probs(emb, gh, gw, T, qi, h0, w0, K)
            top2 = p.topk(2, dim=0).values
            gmarg.append(float((top2[0] - top2[1]).mean()))
            probs.append(p)
            preds.append(p.argmax(0).cpu().numpy())
        for c in range(C):
            single[c].update(preds[c], gt)
        preds_np = np.stack(preds)
        gt_r = np.array(Image.fromarray(gt.astype(np.int32), mode="I").resize(
            (w0, h0), Image.NEAREST))
        valid = (gt_r != ignore) & (gt_r >= 0) & (gt_r < K)
        dfeat = dino_features(dino, img_r, h0, w0, model.device)
        masks = gen.generate(np.array(img))
        reg = build_region_map(masks, h0, w0)
        base = int(np.argmax(gmarg))  # label-free fallback config
        outs_arm = {k: preds_np[base].copy()
                    for k in ["consist", "consist_dino", "margin_ctl", "oracle"]}
        # first pass: crispness for median threshold within image
        crisp = {}
        for r in range(reg.max() + 1):
            sel_np = reg == r
            if sel_np.any():
                crisp[r] = region_crispness(dfeat, sel_np)
        med = float(np.median(list(crisp.values()))) if crisp else 0.0
        crisp_all.extend(crisp.values())
        for r in range(reg.max() + 1):
            sel_np = reg == r
            if not sel_np.any():
                continue
            sel = torch.from_numpy(sel_np).to(model.device)
            rp = [probs[c][:, sel].mean(1) for c in range(C)]
            labs = [int(x.argmax()) for x in rp]
            vals, cnts = np.unique(labs, return_counts=True)
            maj = int(vals[cnts.argmax()])
            agree = [c for c in range(C) if labs[c] == maj]
            c_cons = max(agree, key=lambda c: float(rp[c][maj]))
            margins = [float(x.topk(2).values.diff().abs()) for x in rp]
            c_marg = int(np.argmax(margins))
            v = valid & sel_np
            if v.sum() > 0:
                accs = [float(((preds_np[c] == gt_r) & v).sum() / v.sum())
                        for c in range(C)]
                c_orc = int(np.argmax(accs))
                outs_arm["oracle"][sel_np] = preds_np[c_orc][sel_np]
                n_regions += 1
                match_consist += int(c_cons == c_orc)
                match_margin += int(c_marg == c_orc)
            outs_arm["consist"][sel_np] = preds_np[c_cons][sel_np]
            if crisp.get(r, 0.0) >= med:
                outs_arm["consist_dino"][sel_np] = preds_np[c_cons][sel_np]
            outs_arm["margin_ctl"][sel_np] = preds_np[c_marg][sel_np]
        for k in meters:
            meters[k].update(outs_arm[k], gt_r)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}] " + " ".join(
                f"{k}={meters[k].miou()[0]*100:.2f}" for k in meters), flush=True)

    res = {"prereg": "stage12_month/prereg_w2d_selfroute.md",
           "dataset": a.dataset, "offset": a.offset, "limit": a.limit,
           "pool": [f"{f}_L{L}" for f, L in pool],
           "single_miou": {f"{f}_L{L}": m.miou()[0] * 100
                           for (f, L), m in zip(pool, single)},
           "arms": {k: meters[k].miou()[0] * 100 for k in meters},
           "n_regions": n_regions,
           "match_rate_consist": match_consist / max(n_regions, 1),
           "match_rate_margin": match_margin / max(n_regions, 1),
           "random_match": 1.0 / C,
           "crisp_mean": float(np.mean(crisp_all)) if crisp_all else 0.0}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
