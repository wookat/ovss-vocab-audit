"""prereg_v6: stage-2 region-level visual veto with DINOv2.

Stage 1: SAM region pooled argmax (VABS vocab).
Stage 2: background-won regions whose DINOv2 region embedding is >= theta cosine
to a same-image high-confidence target anchor get flipped to that target.
Control: flip the same number of randomly chosen background regions per image.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from probe_d1sam import build_region_map

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def dino_dense(dino, img_pil, device):
    w, h = img_pil.size
    s = 518 / min(w, h)
    nw, nh = int(round(w * s) // 14 * 14), int(round(h * s) // 14 * 14)
    im = img_pil.resize((nw, nh), Image.BILINEAR)
    a = torch.from_numpy(np.asarray(im).copy()).float().div_(255.0)
    a = (a - torch.tensor(MEAN)) / torch.tensor(STD)
    a = a.permute(2, 0, 1).unsqueeze(0).to(device)
    f = dino.forward_features(a)["x_norm_patchtokens"][0]  # (N, D)
    gh, gw = nh // 14, nw // 14
    f = f.reshape(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)
    f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)[0]
    return F.normalize(f, dim=0)  # (D, h, w)


def region_mean(feat, reg, R):
    D = feat.shape[0]
    flat = feat.reshape(D, -1)
    r = torch.from_numpy(reg.reshape(-1)).to(feat.device)
    valid = r >= 0
    out = torch.zeros(R, D, device=feat.device)
    cnt = torch.zeros(R, device=feat.device)
    out.index_add_(0, r[valid], flat[:, valid].T)
    cnt.index_add_(0, r[valid], torch.ones(int(valid.sum()), device=feat.device))
    return F.normalize(out / cnt.clamp_min(1).unsqueeze(1), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vabs-vocab", required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--thetas", default="0.5,0.6,0.7")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    thetas = [float(x) for x in a.thetas.split(",")]

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to("cuda").eval()

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)

    vabs_names = json.load(open(a.vabs_vocab))
    T_vabs, qi_vabs = class_embeddings(model, vabs_names)
    T_vabs = T_vabs.to(model.device)
    scale = 40.0
    rng = np.random.RandomState(0)

    meters = {"stage1": IoUMeter(K, ignore)}
    for th in thetas:
        meters[f"veto{th}"] = IoUMeter(K, ignore)
        meters[f"randflip{th}"] = IoUMeter(K, ignore)

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T_vabs)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        probs = (scale * sims).softmax(0)
        pooled = torch.zeros(K, h0, w0, device=probs.device)
        idx = qi_vabs.to(probs.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)

        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        R = max(int(reg.max()) + 1, 1)
        # region-avg class probs
        flat = pooled.reshape(K, -1)
        r = torch.from_numpy(reg.reshape(-1)).to(probs.device)
        valid = r >= 0
        cp = torch.zeros(R, K, device=probs.device)
        cnt = torch.zeros(R, device=probs.device)
        cp.index_add_(0, r[valid], flat[:, valid].T)
        cnt.index_add_(0, r[valid], torch.ones(int(valid.sum()), device=probs.device))
        cp = cp / cnt.clamp_min(1).unsqueeze(1)
        reg_pred = cp.argmax(1).cpu().numpy()

        pix_pred = pooled.argmax(0).cpu().numpy()
        pred1 = pix_pred.copy()
        cov = reg >= 0
        pred1[cov] = reg_pred[reg[cov]]
        meters["stage1"].update(pred1, gt)

        dfeat = dino_dense(dino, img.resize((w0, h0)), "cuda")
        remb = region_mean(dfeat, reg, R)  # (R, D)

        cp_np = cp.cpu().numpy()
        for th in thetas:
            pred_v = reg_pred.copy()
            # anchors: for each class c>0, regions predicted c with prob in top 50%
            anchors = {}
            for c in range(1, K):
                rs = np.where(reg_pred == c)[0]
                if len(rs) == 0:
                    continue
                med = np.median(cp_np[rs, c])
                anchors[c] = rs[cp_np[rs, c] >= med]
            bg_regions = np.where(reg_pred == 0)[0]
            flipped = []
            for rr in bg_regions:
                best_c, best_s = 0, -1
                for c, ancs in anchors.items():
                    if len(ancs) == 0:
                        continue
                    s = float((remb[rr] @ remb[ancs].T).max())
                    if s > best_s:
                        best_s, best_c = s, c
                if best_s >= th and best_c > 0:
                    pred_v[rr] = best_c
                    flipped.append(rr)
            pred = pix_pred.copy()
            pred[cov] = pred_v[reg[cov]]
            meters[f"veto{th}"].update(pred, gt)

            # control: flip equal number of random bg regions to random anchor class
            pred_c = reg_pred.copy()
            if flipped and len(anchors):
                sel = rng.choice(bg_regions, size=len(flipped), replace=False)
                cls = rng.choice(list(anchors.keys()), size=len(flipped))
                pred_c[sel] = cls
            pred = pix_pred.copy()
            pred[cov] = pred_c[reg[cov]]
            meters[f"randflip{th}"].update(pred, gt)

        if (i + 1) % 10 == 0:
            msg = " ".join(f"{k}={m.miou()[0]*100:.2f}" for k, m in meters.items())
            print(f"[{i+1}] {msg}", flush=True)

    res = {"prereg": "prereg_v6.md", "variant": a.variant, "dataset": a.dataset,
           "offset": a.offset, "limit": a.limit, "arms": {}}
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
