"""W12-K2 (prereg_w12_k2_contamination.md): contamination curves —
GT-present and all-class mIoU vs number of injected absent entries."""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import (IoUMeter, class_embeddings, resize_short, to_tensor,
                      seg_logits)

SIZES = [0, 10, 25, 50, 100, 150]


def build_pools():
    voc = json.load(open("perturbed_vocabs/voc21_plain.json"))
    near_full = json.load(open("perturbed_vocabs/voc21_dis_near200.json"))
    pool_near = near_full[len(voc):][:150]
    ade = json.load(open("perturbed_vocabs/ade150_plain.json"))
    ctx = json.load(open("perturbed_vocabs/ctx60_plain.json"))
    vocset = set(w.strip().lower() for n in voc for w in n.split(","))
    cand = []
    for n in ade[1:] + ctx[1:]:
        w = n.split(",")[0].strip().lower()
        if w and w not in vocset and w not in cand:
            cand.append(w)
    rng = random.Random(0)
    rng.shuffle(cand)
    pool_rand = cand[:150]
    return voc, {"near": pool_near, "rand": pool_rand}


@torch.no_grad()
def run(variant, out_path, limit=300):
    samples, _, ignore = data.DATASETS["voc21"]()
    voc, pools = build_pools()
    res = {"prereg": "prereg_w12_k2_contamination.md", "variant": variant}
    model = DenseCLIP(variant, device="cuda")
    for pname, pool in pools.items():
        names = voc + pool
        emb, _ = class_embeddings(model, names, "none")
        emb = emb.to(model.device)
        meters = {n: (IoUMeter(len(voc) + n, ignore)) for n in SIZES}
        for ip, gp, loader in samples[:limit]:
            gt = loader(gp)
            img = Image.open(ip).convert("RGB")
            img_r, (w0, h0) = resize_short(img, 336)
            t = to_tensor(img_r, model.device)
            logits = seg_logits(model, t, emb, 224, 112)
            logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0),
                                   mode="bilinear", align_corners=False)[0]
            for n in SIZES:
                sub = logits[:len(voc) + n]
                pred = sub.argmax(0).cpu().numpy().astype(np.int64)
                meters[n].update(pred, gt)
        res[pname] = {str(n): {"gt_present": meters[n].miou()[0] * 100,
                               "all_class": meters[n].miou_all() * 100}
                      for n in SIZES}
        print(pname, {n: round(res[pname][str(n)]["gt_present"], 2)
                      for n in SIZES}, flush=True)
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.variant, a.out)
