"""Qualitative visualizations for the vocabulary audit paper.

Fig A (qual grid): image | GT | predictions under official / plain / syn100 / hypernym / plain+200near.
Fig B (distractor absorption): plain vs plain+200near, distractor-predicted pixels highlighted in red.
"""
import json, os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor
import data

OUT = "/media/dell/DATA/ovss/figs"
os.makedirs(OUT, exist_ok=True)

VOC_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128],
], dtype=np.uint8)


def colorize(pred, K=21):
    """pred (H,W) ints; classes >= K (distractors) -> red."""
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    m = pred < K
    out[m] = VOC_COLORS[pred[m] % 21]
    out[~m] = [255, 40, 40]
    return out


@torch.no_grad()
def predict(model, img_pil, names, short=336, logit_scale=40.0):
    text_emb, qidx = class_embeddings(model, names, "none")
    text_emb = text_emb.to(model.device)
    K = len(names)
    img_r, (w0, h0) = resize_short(img_pil, short)
    t = to_tensor(img_r, model.device)
    logits = seg_logits(model, t, text_emb)
    logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False)[0]
    probs = (logit_scale * logits).softmax(0)
    if text_emb.shape[0] != K:
        pooled = torch.zeros(K, *probs.shape[1:], device=probs.device)
        idx = qidx.to(model.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
        probs = pooled
    return probs.argmax(0).cpu().numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--images", type=int, nargs="+", default=[3, 17, 42])
    a = ap.parse_args()
    model = DenseCLIP(a.variant, device="cuda")
    samples, official, ignore = data.DATASETS["voc21"]()
    pv = "perturbed_vocabs"
    vocabs = {
        "official": official,
        "plain": json.load(open(f"{pv}/voc21_plain.json")),
        "syn100": json.load(open(f"{pv}/voc21_syn100_s0.json")),
        "hypernym": json.load(open(f"{pv}/voc21_gran_coarse.json")),
        "dis200": json.load(open(f"{pv}/voc21_dis_near200.json")),
    }
    for i in a.images:
        img_path, gt_path, gt_loader = samples[i]
        img = Image.open(img_path).convert("RGB")
        gt = gt_loader(gt_path)
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        m = gt != ignore
        gt_vis[m] = VOC_COLORS[gt[m] % 21]
        gt_vis[~m] = [255, 255, 255]
        base = os.path.basename(img_path).split(".")[0]
        img.save(f"{OUT}/{base}_img.png")
        Image.fromarray(gt_vis).save(f"{OUT}/{base}_gt.png")
        for vn, names in vocabs.items():
            pred = predict(model, img, names)
            Image.fromarray(colorize(pred)).save(f"{OUT}/{base}_{a.variant}_{vn}.png")
            print(f"{base} {vn} done", flush=True)


if __name__ == "__main__":
    main()
