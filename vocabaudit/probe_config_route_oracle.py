"""C6 per-region config-routing oracle (stage11_phase3/prereg_c2c6_oracle.md).

20 configs = 4 flavors x exit layers 8..12. Per image: predictions from every
config; SAM regions; oracle = per-region best config by pixel accuracy vs GT.
Also per-image oracle and each config's own mIoU.
"""
import argparse, json
import numpy as np
import torch
from PIL import Image

import data
from probe_headlens import seg_pred, FLAVORS
from probe_headlens_e2 import HeadLensL
from probe_d1sam import build_region_map
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vocab", default="perturbed_vocabs/voc21_official.json")
    ap.add_argument("--layers", default="8,9,10,11,12")
    ap.add_argument("--sam-ckpt", default="/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points_per_side)

    model = HeadLensL()
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    layers = [int(x) for x in a.layers.split(",")]
    cfgs = [(f, L) for f in FLAVORS for L in layers]
    keys = [f"{f}_L{L}" for f, L in cfgs]

    meters = {k: IoUMeter(K, ignore) for k in keys}
    m_reg_oracle = IoUMeter(K, ignore)
    m_img_oracle = IoUMeter(K, ignore)
    Hn = 12

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        preds = []
        for (f, L), k in zip(cfgs, keys):
            outs, bias, (gh, gw) = model.head_outputs_at(t, f, L)
            emb = model.subset_emb(outs, bias, range(Hn))
            pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
            meters[k].update(pred, gt)
            preds.append(pred)
        preds = np.stack(preds)  # (C,h,w)

        gt_r = np.array(Image.fromarray(gt.astype(np.int32), mode="I").resize(
            (w0, h0), Image.NEAREST))
        valid = (gt_r != ignore) & (gt_r >= 0) & (gt_r < K)
        correct = (preds == gt_r[None]) & valid[None]  # (C,h,w)

        # per-image oracle
        best_img = int(correct.reshape(len(cfgs), -1).sum(1).argmax())
        m_img_oracle.update(preds[best_img], gt_r)

        # per-region oracle: SAM regions; uncovered pixels use per-image best cfg
        masks = gen.generate(np.array(img))
        reg = build_region_map(masks, h0, w0)
        out = preds[best_img].copy()
        for r in range(reg.max() + 1):
            sel = reg == r
            if not sel.any():
                continue
            acc = correct[:, sel].sum(1)
            out[sel] = preds[int(acc.argmax())][sel]
        m_reg_oracle.update(out, gt_r)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}] reg_oracle={m_reg_oracle.miou()[0]*100:.2f} "
                  f"img_oracle={m_img_oracle.miou()[0]*100:.2f}", flush=True)

    res = {"prereg": "stage11_phase3/prereg_c2c6_oracle.md",
           "configs": {k: meters[k].miou()[0] * 100 for k in keys},
           "region_oracle_miou": m_reg_oracle.miou()[0] * 100,
           "image_oracle_miou": m_img_oracle.miou()[0] * 100}
    res["best_single"] = max(res["configs"].values())
    res["region_oracle_gain"] = res["region_oracle_miou"] - res["best_single"]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "configs"}, indent=1))


if __name__ == "__main__":
    main()
