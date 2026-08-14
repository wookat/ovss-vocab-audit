"""W13-L3 (prereg_w13_l3_filler.md): filler free-lunch mechanism —
pseudo-word arm, GT-row decomposition, background-boost control."""
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

BOOSTS = (0.01, 0.02, 0.05)


def pseudo_words(n, seed=0):
    rng = random.Random(seed)
    cons = "bcdfgklmnprstvz"
    vow = "aeiou"
    out = []
    while len(out) < n:
        w = "".join(rng.choice(cons) + rng.choice(vow)
                    for _ in range(rng.randint(2, 3)))
        if w not in out:
            out.append(w)
    return out


@torch.no_grad()
def run(out_path, variant="sclip", limit=300):
    samples, _, ignore = data.DATASETS["voc21"]()
    voc = json.load(open("perturbed_vocabs/voc21_plain.json"))
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
    vv = json.load(open("perturbed_vocabs/voc21_plain_vabs64.json"))
    vabs50 = vv[len(voc):][:50]

    pools = {"R": cand[:50], "P": pseudo_words(50)}
    if vabs50:
        pools["V"] = vabs50

    model = DenseCLIP(variant, device="cuda")
    res = {"prereg": "prereg_w13_l3_filler.md", "variant": variant}

    emb0, _ = class_embeddings(model, voc, "none")
    emb0 = emb0.to(model.device)
    K0 = len(voc)

    arm_embs = {}
    for a, pool in pools.items():
        e, _ = class_embeddings(model, voc + pool, "none")
        arm_embs[a] = e.to(model.device)

    m_plain = IoUMeter(K0, ignore)
    m_boost = {b: IoUMeter(K0, ignore) for b in BOOSTS}
    m_arm = {a: IoUMeter(K0 + 50, ignore) for a in pools}
    filler_bg = {a: 0 for a in pools}
    filler_fg = {a: 0 for a in pools}

    for ip, gp, loader in samples[:limit]:
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        lg0 = seg_logits(model, t, emb0, 224, 112)
        lg0 = F.interpolate(lg0.unsqueeze(0), size=(h0, w0),
                            mode="bilinear", align_corners=False)[0]
        m_plain.update(lg0.argmax(0).cpu().numpy().astype(np.int64), gt)
        for b in BOOSTS:
            lb = lg0.clone()
            lb[0] += b
            m_boost[b].update(lb.argmax(0).cpu().numpy().astype(np.int64), gt)
        for a in pools:
            lga = seg_logits(model, t, arm_embs[a], 224, 112)
            lga = F.interpolate(lga.unsqueeze(0), size=(h0, w0),
                                mode="bilinear", align_corners=False)[0]
            pred = lga.argmax(0).cpu().numpy().astype(np.int64)
            m_arm[a].update(pred, gt)
            fm = (pred >= K0) & (gt != ignore)
            filler_bg[a] += int(((gt == 0) & fm).sum())
            filler_fg[a] += int(((gt > 0) & fm).sum())

    res["plain"] = m_plain.miou()[0] * 100
    res["boost"] = {str(b): m_boost[b].miou()[0] * 100 for b in BOOSTS}
    for a in pools:
        tot = filler_bg[a] + filler_fg[a]
        res[a] = {"gt_present": m_arm[a].miou()[0] * 100,
                  "filler_bg_share": filler_bg[a] / max(tot, 1),
                  "filler_pixels": tot}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="sclip")
    a = ap.parse_args()
    run(a.out, a.variant)
