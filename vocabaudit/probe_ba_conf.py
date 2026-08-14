"""W5e step 1: confusion-matrix capture (prereg_w5e_ba.md, frozen).

Runs one (method, vocab) cell under the unified protocol and stores the full
GT-row confusion matrix C[gt in 0..Kg-1, pred in 0..Kv-1] plus per-class
inter/union, for offline flow decomposition by probe_ba_decomp.py.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="clearclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--n-gt-classes", type=int, required=True)
    ap.add_argument("--out", required=True, help=".npz output path")
    a = ap.parse_args()

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
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    qi = qi.to(model.device)
    Kv = len(names)
    Kg = a.n_gt_classes
    scale = 40.0
    meter = IoUMeter(Kv, ignore)
    conf = np.zeros((Kg, Kv), dtype=np.int64)

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
        probs.scatter_reduce_(0, qi.view(-1, 1, 1).expand_as(probs_q), probs_q,
                              reduce="amax", include_self=False)
        pred = probs.argmax(0).cpu().numpy().astype(np.int64)
        gta = np.asarray(gt)
        m = gta != ignore
        meter.update(pred, gta)
        k = gta[m].astype(np.int64) * Kv + pred[m]
        conf += np.bincount(k, minlength=Kg * Kv)[:Kg * Kv].reshape(Kg, Kv)
        if (i + 1) % 50 == 0:
            print(f"[{i+1}] mIoU={meter.miou()[0]*100:.2f}", flush=True)

    miou_gt, _ = meter.miou()
    np.savez_compressed(a.out, conf=conf, inter=meter.inter, union=meter.union,
                        seen=meter.seen,
                        miou_gt=miou_gt, miou_all=meter.miou_all(),
                        kg=Kg, kv=Kv)
    print(json.dumps({"variant": a.variant, "vocab": a.vocab_file,
                      "miou_gt": round(miou_gt * 100, 2),
                      "miou_all": round(meter.miou_all() * 100, 2)}))


if __name__ == "__main__":
    main()
