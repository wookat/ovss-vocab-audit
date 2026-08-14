"""W10-H4 SCLIP arms (prereg_w10_h4_zhinterface.md): bare-name Chinese
queries (no English template) and 2-char short forms."""
import argparse
import json

import torch
import torch.nn.functional as F

import data
from clip_seg import DenseCLIP
from eval_seg import IoUMeter, resize_short, to_tensor, seg_logits

import numpy as np
from PIL import Image


@torch.no_grad()
def evaluate_bare(model, samples, names, limit, ignore):
    emb = model.encode_text_raw(names)
    emb = F.normalize(emb.float(), dim=-1).to(model.device)
    K = len(names)
    meter = IoUMeter(K, ignore)
    for ip, gp, loader in samples[:limit]:
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        logits = seg_logits(model, t, emb, 224, 112)
        logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0),
                               mode="bilinear", align_corners=False)[0]
        pred = logits.argmax(0).cpu().numpy().astype(np.int64)
        meter.update(pred, loader(gp))
    return meter.miou()[0] * 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, _, ignore = data.DATASETS["voc21"]()
    model = DenseCLIP("sclip", device="cuda")
    zh = [n.split(",")[0].strip()
          for n in json.load(open("perturbed_vocabs/voc21_zh.json"))]
    short = [n[:2] for n in zh]
    res = {"prereg": "prereg_w10_h4_zhinterface.md",
           "a1_bare": evaluate_bare(model, samples, zh, 300, ignore),
           "a2_short_bare": evaluate_bare(model, samples, short, 300,
                                          ignore)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
