"""External anchor baseline: pixel-level prediction with PAMR refinement.

Reviewer-requested anchor row (e.g. NACLIP + PAMR on the official vocabulary),
same unified protocol otherwise (short side 336, window 224/112, scale 40).
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from pamr import PAMR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="naclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--skip-dev", action="store_true")
    ap.add_argument("--pamr-iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = DenseCLIP(a.variant)
    pamr = PAMR(num_iter=a.pamr_iters).to(model.device)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    if a.skip_dev:
        samples = samples[:300] + samples[400:]
    K = len(plain_names)

    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    meters = {k: IoUMeter(K, ignore) for k in ["pix", "pix_pamr"]}
    scale = 40.0
    t_pamr = 0.0

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        k_out = int(qi.max()) + 1
        probs = (scale * sims).softmax(0)
        pooled = torch.zeros(k_out, *sims.shape[1:], device=sims.device)
        idx = qi.to(sims.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
        meters["pix"].update(pooled.argmax(0).cpu().numpy(), gt)

        t0 = time.time()
        img_t = to_tensor(img.resize((w0, h0)), model.device)
        refined = pamr(img_t.unsqueeze(0) if img_t.dim() == 3 else img_t,
                       pooled.unsqueeze(0))[0]
        torch.cuda.synchronize(); t_pamr += time.time() - t0
        meters["pix_pamr"].update(refined.argmax(0).cpu().numpy(), gt)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}] pix={meters['pix'].miou()[0]*100:.2f} "
                  f"pamr={meters['pix_pamr'].miou()[0]*100:.2f}", flush=True)

    res = {"variant": a.variant, "dataset": a.dataset, "vocab": a.vocab,
           "skip_dev": bool(a.skip_dev), "pamr_iters": a.pamr_iters,
           "timing_s_per_img": {"pamr": t_pamr / max(len(samples), 1)}, "arms": {}}
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "miou_all": m.miou_all(), "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
