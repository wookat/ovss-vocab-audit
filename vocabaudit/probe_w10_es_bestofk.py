"""W10 addendum (prereg_w10_es_bestofk.md): Spanish best-of-3 translation
control. SCLIP per-vocab per-class IoU; best-of-3 = per-class max."""
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


@torch.no_grad()
def per_class_iou(model, samples, names, limit, ignore):
    emb, _ = class_embeddings(model, names, "none")
    emb = emb.to(model.device)
    K = len(names)
    meter = IoUMeter(K, ignore)
    for ip, gp, loader in samples[:limit]:
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        logits = seg_logits(model, t, emb, 224, 112)
        logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0),
                               mode="bilinear", align_corners=False)[0]
        pred = (40.0 * logits).softmax(0).argmax(0).cpu().numpy()
        meter.update(pred.astype(np.int64), loader(gp))
    iou = meter.inter / np.maximum(meter.union, 1)
    present = meter.union > 0
    return iou, present


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, _, ignore = data.DATASETS["voc21"]()
    model = DenseCLIP("sclip", device="cuda")
    res = {"prereg": "prereg_w10_es_bestofk.md"}
    ious = []
    for tag in ["es", "es_alt1", "es_alt2"]:
        names = json.load(open(f"perturbed_vocabs/voc21_{tag}.json"))
        iou, present = per_class_iou(model, samples, names, 300, ignore)
        res[tag] = float(iou[present].mean() * 100)
        ious.append(iou)
        print(tag, res[tag], flush=True)
    best = np.maximum.reduce(ious)
    res["best_of_3"] = float(best[present].mean() * 100)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
