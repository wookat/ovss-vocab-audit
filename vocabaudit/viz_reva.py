"""Qualitative figure for the REVA paper.

Grid: image | GT | plain (pixel) | plain+VABS (pixel) | REVA (VABS+SAM) | official (pixel)
Rows: selected VOC val images (successes + honest person/tvmonitor failure cases).
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor
from probe_d1sam import build_region_map, pool_regions

OUT = "/media/dell/DATA/ovss/figs"
os.makedirs(OUT, exist_ok=True)

VOC_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128],
], dtype=np.uint8)


def colorize(pred):
    return VOC_COLORS[np.clip(pred, 0, 20)]


@torch.no_grad()
def class_probs(model, t, T, qi, h0, w0, K, scale=40.0):
    sims = seg_logits(model, t, T)
    sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                         align_corners=False)[0]
    probs = (scale * sims).softmax(0)
    pooled = torch.zeros(K, *sims.shape[1:], device=sims.device)
    idx = qi.to(sims.device).view(-1, 1, 1).expand_as(probs)
    pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="naclip")
    ap.add_argument("--indices", default="10,45,120,510")
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--out", default="reva_qual.png")
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    K = len(plain_names)

    vocabs = {}
    for key, path in [("plain", "perturbed_vocabs/voc21_plain.json"),
                      ("vabs", "perturbed_vocabs/voc21_plain_vabs64.json"),
                      ("official", "perturbed_vocabs/voc21_official.json")]:
        names = json.load(open(path))
        T, qi = class_embeddings(model, names)
        vocabs[key] = (T.to(model.device), qi)

    idxs = [int(x) for x in a.indices.split(",")]
    rows = []
    for ix in idxs:
        ip, gp, loader = samples[ix]
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)

        p_plain = class_probs(model, t, *vocabs["plain"], h0, w0, K)
        p_vabs = class_probs(model, t, *vocabs["vabs"], h0, w0, K)
        p_off = class_probs(model, t, *vocabs["official"], h0, w0, K)

        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        reva = pool_regions(p_vabs, reg)

        gt_vis = gt.copy()
        gt_vis[gt_vis == 255] = 0
        row = [np.asarray(img.resize((w0, h0))), colorize(gt_vis),
               colorize(p_plain.argmax(0).cpu().numpy()),
               colorize(p_vabs.argmax(0).cpu().numpy()),
               colorize(reva), colorize(p_off.argmax(0).cpu().numpy())]
        # center-crop to 4:3 then resize to fixed cell
        CH, CW = 168, 224
        def cell(x):
            h, w = x.shape[:2]
            ar = CW / CH
            if w / h > ar:
                nw = int(h * ar); x = x[:, (w - nw) // 2:(w - nw) // 2 + nw]
            else:
                nh = int(w / ar); x = x[(h - nh) // 2:(h - nh) // 2 + nh]
            return np.asarray(Image.fromarray(x).resize((CW, CH), Image.NEAREST))
        row = [cell(x) for x in row]
        rows.append(np.concatenate([np.pad(x, ((0, 0), (0, 3), (0, 0)),
                                           constant_values=255) for x in row], axis=1))

    rows = [np.pad(r, ((0, 3), (0, 0), (0, 0)), constant_values=255) for r in rows]
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(os.path.join(OUT, a.out))
    print("saved", os.path.join(OUT, a.out))


if __name__ == "__main__":
    main()
