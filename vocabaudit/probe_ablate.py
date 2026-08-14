"""Scoop-check mandated same-protocol ablations (VOC test-300):

- featpool_vabs : LaVG-style feature pooling inside SAM masks (pool dense CLIP
  features, then text argmax) with the VABS vocabulary -- vs our prob pooling.
- sam_handbg    : our probability pooling with plain vocab + hand-written
  background sub-queries (SCLIP official background expansion) instead of VABS.
- pix_handbg    : pixel-level hand-written background arm.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from probe_d1sam import build_region_map, pool_regions


@torch.no_grad()
def dense_feats(model, img, T_dim, window=224, stride=112):
    _, _, H, W = img.shape
    out = torch.zeros(T_dim, H, W, device=img.device)
    cnt = torch.zeros(1, H, W, device=img.device)
    hs = list(range(0, max(H - window, 0) + 1, stride))
    ws = list(range(0, max(W - window, 0) + 1, stride))
    if hs[-1] + window < H: hs.append(H - window)
    if ws[-1] + window < W: ws.append(W - window)
    for y in hs:
        for x in ws:
            crop = img[:, :, y:y+window, x:x+window]
            feat, (gh, gw) = model.encode_dense(crop)
            feat = feat[0].float().reshape(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)
            feat = F.interpolate(feat, size=(window, window), mode="bilinear",
                                 align_corners=False)
            out[:, y:y+window, x:x+window] += feat[0]
            cnt[:, y:y+window, x:x+window] += 1
    return out / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vabs-vocab", required=True)
    ap.add_argument("--handbg-vocab", required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--skip-dev", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    if a.skip_dev:
        samples = samples[:300] + samples[400:]
    K = len(plain_names)

    vabs_names = json.load(open(a.vabs_vocab))
    hand_names = json.load(open(a.handbg_vocab))
    T_vabs, qi_vabs = class_embeddings(model, vabs_names)
    T_hand, qi_hand = class_embeddings(model, hand_names)
    T_vabs, T_hand = T_vabs.to(model.device), T_hand.to(model.device)
    D = T_vabs.shape[1]
    scale = 40.0

    meters = {k: IoUMeter(K, ignore) for k in
              ["featpool_vabs", "sam_handbg", "pix_handbg"]}

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)

        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)

        # LaVG-style: pooled features -> cosine -> argmax over VABS queries
        feats = dense_feats(model, t, D)
        feats = F.interpolate(feats.unsqueeze(0), size=(h0, w0), mode="bilinear",
                              align_corners=False)[0]
        featsn = F.normalize(feats, dim=0)
        flat = featsn.reshape(D, -1)
        r = torch.from_numpy(reg.reshape(-1)).to(model.device)
        valid = r >= 0
        R = max(int(reg.max()) + 1, 1)
        fr = torch.zeros(R, D, device=model.device)
        cnt = torch.zeros(R, device=model.device)
        fr.index_add_(0, r[valid], flat[:, valid].T)
        cnt.index_add_(0, r[valid], torch.ones(int(valid.sum()), device=model.device))
        fr = F.normalize(fr / cnt.clamp_min(1).unsqueeze(1), dim=1)
        qsim = fr @ T_vabs.T  # (R, Q)
        qi = qi_vabs.to(model.device)
        csim = torch.full((R, K), -1e9, device=model.device)
        for c in range(K):
            cols = (qi == c).nonzero().squeeze(1)
            csim[:, c] = qsim[:, cols].max(1).values
        reg_pred = csim.argmax(1).cpu().numpy()
        # pixel fallback: pixel-level cosine argmax
        pixsim = (featsn.reshape(D, -1).T @ T_vabs.T)
        pixc = torch.full((pixsim.shape[0], K), -1e9, device=model.device)
        for c in range(K):
            cols = (qi == c).nonzero().squeeze(1)
            pixc[:, c] = pixsim[:, cols].max(1).values
        pix_pred = pixc.argmax(1).cpu().numpy().reshape(h0, w0)
        pred = pix_pred.copy()
        cov = reg >= 0
        pred[cov] = reg_pred[reg[cov]]
        meters["featpool_vabs"].update(pred, gt)

        # hand-written background arms (prob pooling, same as REVA)
        sims = seg_logits(model, t, T_hand)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        probs = (scale * sims).softmax(0)
        pooled = torch.zeros(K, h0, w0, device=probs.device)
        idxh = qi_hand.to(probs.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idxh, probs, reduce="amax", include_self=False)
        meters["pix_handbg"].update(pooled.argmax(0).cpu().numpy(), gt)
        meters["sam_handbg"].update(pool_regions(pooled, reg), gt)

        if (i + 1) % 20 == 0:
            msg = " ".join(f"{k}={m.miou()[0]*100:.2f}" for k, m in meters.items())
            print(f"[{i+1}] {msg}", flush=True)

    res = {"variant": a.variant, "dataset": a.dataset, "offset": a.offset,
           "limit": a.limit, "arms": {}}
    res["skip_dev"] = bool(a.skip_dev)
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "miou_all": m.miou_all(), "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
