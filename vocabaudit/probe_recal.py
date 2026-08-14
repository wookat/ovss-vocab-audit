"""W6-F1 RECAL (prereg_w6f1_recal.md, frozen): transductive name-conditioned
logit debiasing for the synonym axis.

Pass A caches class-pooled cosine-sim maps per image (float16, disk). The EM
loop and the final biased evaluation run from the cache. Bias (frozen):
b_c = (log p_ref - log p_hat) / 40 on the cosine scale, p_ref prop. to
p_hat^0.5, background excluded from re-balancing, 3 iterations.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter

ALPHA = 0.5
N_ITERS = 3
SCALE = 40.0
EPS = 1e-8


def build_cache(variant, dataset, offset, limit, vocab_file, cache_dir):
    model = DenseCLIP(variant)
    samples, _, ignore = data.DATASETS[dataset]()
    samples = samples[offset:offset + limit]
    names = json.load(open(vocab_file))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    qi = qi.to(model.device)
    Kv = len(names)
    os.makedirs(cache_dir, exist_ok=True)
    for i, (ip, gp, loader) in enumerate(samples):
        fp = os.path.join(cache_dir, f"{i:04d}.npz")
        if os.path.exists(fp):
            continue
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        pooled = torch.full((Kv, h0, w0), -1e4, device=sims.device)
        pooled.scatter_reduce_(0, qi.view(-1, 1, 1).expand_as(sims), sims,
                               reduce="amax", include_self=True)
        np.savez_compressed(fp, sims=pooled.half().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"cache [{i+1}/{len(samples)}]", flush=True)
    return samples, ignore, Kv


def mass(cache_dir, n, Kv, b):
    cnt = np.zeros(Kv)
    for i in range(n):
        sims = np.load(os.path.join(cache_dir, f"{i:04d}.npz"))["sims"].astype(np.float32)
        pred = (sims + b[:, None, None]).argmax(0)
        cnt += np.bincount(pred.ravel(), minlength=Kv)
    return cnt / cnt.sum()


def evaluate(cache_dir, samples, ignore, Kv, b):
    meter = IoUMeter(Kv, ignore)
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        sims = np.load(os.path.join(cache_dir, f"{i:04d}.npz"))["sims"].astype(np.float32)
        pred = (sims + b[:, None, None]).argmax(0)
        meter.update(pred, np.asarray(gt))
    return meter.miou()[0] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--has-bg", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, ignore, Kv = build_cache(a.variant, a.dataset, a.offset, a.limit,
                                      a.vocab_file, a.cache_dir)
    n = len(samples)
    b = np.zeros(Kv)
    hist = []
    for it in range(N_ITERS):
        p_hat = mass(a.cache_dir, n, Kv, b)
        fg = np.arange(Kv) >= (1 if a.has_bg else 0)
        p = np.clip(p_hat[fg], EPS, None)
        p_ref = p ** ALPHA
        p_ref /= p_ref.sum()
        p_n = p / p.sum()
        delta = np.zeros(Kv)
        delta[fg] = (np.log(p_ref) - np.log(p_n)) / SCALE
        b = b + delta
        hist.append({"iter": it, "max_abs_b": float(np.abs(b).max())})
        print(f"iter {it} max|b|={np.abs(b).max():.4f}", flush=True)

    miou_base = evaluate(a.cache_dir, samples, ignore, Kv, np.zeros(Kv))
    miou_recal = evaluate(a.cache_dir, samples, ignore, Kv, b)
    res = {"prereg": "prereg_w6f1_recal.md", "variant": a.variant,
           "vocab": a.vocab_file, "alpha": ALPHA, "iters": N_ITERS,
           "miou_base": miou_base, "miou_recal": miou_recal,
           "bias": b.tolist(), "hist": hist}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"variant": a.variant, "vocab": os.path.basename(a.vocab_file),
                      "base": round(miou_base, 2), "recal": round(miou_recal, 2)}))


if __name__ == "__main__":
    main()
