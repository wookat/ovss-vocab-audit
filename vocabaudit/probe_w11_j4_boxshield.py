"""W11-J4 (prereg_w11_j4_boxshield.md): OWLv2 box-evidence shield over
SCLIP dense predictions on the distractor vocabulary."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

import data
from clip_seg import DenseCLIP
from eval_seg import (IoUMeter, class_embeddings, resize_short, to_tensor,
                      seg_logits)

THRESH = 0.2


@torch.no_grad()
def run(vocab_file, out_path, limit=300):
    samples, _, ignore = data.DATASETS["voc21"]()
    names = json.load(open(vocab_file))
    labels = [n.split(",")[0].strip() for n in names]
    K = len(names)

    model = DenseCLIP("sclip", device="cuda")
    emb, _ = class_embeddings(model, names, "none")
    emb = emb.to(model.device)

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    queries = [f"a photo of a {l}" for l in labels[1:]]

    m_dense = IoUMeter(K, ignore)
    m_shield = IoUMeter(K, ignore)
    for i, (ip, gp, loader) in enumerate(samples[:limit]):
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        logits = seg_logits(model, t, emb, 224, 112)
        logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0),
                               mode="bilinear", align_corners=False)[0]
        pred = logits.argmax(0).cpu().numpy().astype(np.int64)

        inputs = proc(text=[queries], images=img, padding="max_length",
                      return_tensors="pt").to("cuda")
        out = owl(**inputs)
        det = proc.post_process_object_detection(
            out, threshold=THRESH, target_sizes=torch.tensor([(h0, w0)]))[0]
        supported = np.zeros(K, dtype=bool)
        supported[0] = True  # background always allowed
        for li in det["labels"].tolist():
            supported[li + 1] = True

        allowed = torch.from_numpy(supported).to(logits.device)
        lg = logits.clone()
        lg[~allowed] = -1e4
        shield = lg.argmax(0).cpu().numpy().astype(np.int64)

        gt = loader(gp)
        m_dense.update(pred, gt)
        m_shield.update(shield, gt)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}] dense_all={m_dense.miou_all()*100:.2f} "
                  f"shield_all={m_shield.miou_all()*100:.2f}", flush=True)

    res = {"prereg": "prereg_w11_j4_boxshield.md", "vocab": vocab_file,
           "dense_all": m_dense.miou_all() * 100,
           "dense_gt": m_dense.miou()[0] * 100,
           "shield_all": m_shield.miou_all() * 100,
           "shield_gt": m_shield.miou()[0] * 100}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.vocab_file, a.out)
