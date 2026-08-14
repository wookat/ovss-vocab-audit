"""W13-L5 (prereg_w13_l5_vabsboost.md): scalar background boost vs VABS
under matched fold convention, dev-tuned boost."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import (IoUMeter, class_embeddings, resize_short, to_tensor,
                      seg_logits)

BOOSTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)


@torch.no_grad()
def collect(model, emb, qidx, samples, ignore, K):
    """Return list of (per-class pooled logits, gt): amax over sub-queries
    (SCLIP convention; VABS negatives are background sub-queries)."""
    out = []
    qidx = qidx.to(model.device)
    for ip, gp, loader in samples:
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        lg = seg_logits(model, t, emb, 224, 112)
        lg = F.interpolate(lg.unsqueeze(0), size=(h0, w0),
                           mode="bilinear", align_corners=False)[0]
        if lg.shape[0] != K:
            pooled = torch.full((K, *lg.shape[1:]), -1e4, device=lg.device)
            idx = qidx.view(-1, 1, 1).expand_as(lg)
            pooled.scatter_reduce_(0, idx, lg, reduce="amax",
                                   include_self=True)
            lg = pooled
        out.append((lg.cpu().half(), gt))
    return out


def miou_with_boost(cache, K, ignore, boost):
    m = IoUMeter(K, ignore)
    for lg, gt in cache:
        l2 = lg.float()
        l2[0] += boost
        m.update(l2.argmax(0).numpy().astype(np.int64), gt)
    return m.miou()[0] * 100


@torch.no_grad()
def run(variant, out_path):
    samples, _, ignore = data.DATASETS["voc21"]()
    voc = json.load(open("perturbed_vocabs/voc21_plain.json"))
    K = len(voc)
    vabs_names = json.load(open("perturbed_vocabs/voc21_plain_vabs64.json"))
    model = DenseCLIP(variant, device="cuda")
    emb_p, qidx_p = class_embeddings(model, voc, "none")
    emb_v, qidx_v = class_embeddings(model, vabs_names, "none")
    emb_p, emb_v = emb_p.to(model.device), emb_v.to(model.device)

    dev, ev = samples[:100], samples[100:300]
    res = {"prereg": "prereg_w13_l5_vabsboost.md", "variant": variant}
    for arm, emb, qi in (("plain", emb_p, qidx_p), ("vabs", emb_v, qidx_v)):
        cache_dev = collect(model, emb, qi, dev, ignore, K)
        cache_ev = collect(model, emb, qi, ev, ignore, K)
        dev_scores = {b: miou_with_boost(cache_dev, K, ignore, b)
                      for b in (0.0,) + BOOSTS}
        bstar = max(BOOSTS, key=lambda b: dev_scores[b])
        res[arm] = {
            "dev_scores": {str(b): round(v, 3)
                           for b, v in dev_scores.items()},
            "b_star": bstar,
            "eval_noboost": miou_with_boost(cache_ev, K, ignore, 0.0),
            "eval_boost": miou_with_boost(cache_ev, K, ignore, bstar)}
        print(arm, res[arm], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.variant, a.out)
