"""W12-K3 (prereg_w12_k3_oracle.md): GT-presence oracle pruning upper
bound on the distractor vocabulary, three dense models."""
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
def run(variant, vocab_file, out_path, limit=300):
    samples, _, ignore = data.DATASETS["voc21"]()
    names = json.load(open(vocab_file))
    K = len(names)

    model = DenseCLIP(variant, device="cuda")
    emb, _ = class_embeddings(model, names, "none")
    emb = emb.to(model.device)

    m_dense = IoUMeter(K, ignore)
    m_oracle = IoUMeter(K, ignore)
    for i, (ip, gp, loader) in enumerate(samples[:limit]):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        logits = seg_logits(model, t, emb, 224, 112)
        logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0),
                               mode="bilinear", align_corners=False)[0]
        pred = logits.argmax(0).cpu().numpy().astype(np.int64)

        present = np.unique(gt)
        supported = np.zeros(K, dtype=bool)
        supported[0] = True
        for c in present:
            if 0 < c < 21:  # GT-present foreground classes only
                supported[c] = True
        allowed = torch.from_numpy(supported).to(logits.device)
        lg = logits.clone()
        lg[~allowed] = -1e4
        orc = lg.argmax(0).cpu().numpy().astype(np.int64)

        m_dense.update(pred, gt)
        m_oracle.update(orc, gt)
        if (i + 1) % 50 == 0:
            print(f"[{i+1}] dense_all={m_dense.miou_all()*100:.2f} "
                  f"oracle_all={m_oracle.miou_all()*100:.2f}", flush=True)

    res = {"prereg": "prereg_w12_k3_oracle.md", "variant": variant,
           "vocab": vocab_file,
           "dense_all": m_dense.miou_all() * 100,
           "dense_gt": m_dense.miou()[0] * 100,
           "oracle_all": m_oracle.miou_all() * 100,
           "oracle_gt": m_oracle.miou()[0] * 100}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.variant, a.vocab_file, a.out)
