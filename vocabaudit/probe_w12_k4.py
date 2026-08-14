"""W12-K4 (prereg_w12_k4_redteam.md): J5 pruning under the archived ANS
attack (heldout offset=100, limit=200) + ADE-150 scale cells."""
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
def cell(variant, ds, vocab_file, proc, owl, offset=0, limit=300):
    samples, _, ignore = data.DATASETS[ds]()
    names = json.load(open(vocab_file))
    labels = [n.split(",")[0].strip() for n in names]
    K = len(names)
    model = DenseCLIP(variant, device="cuda")
    emb, _ = class_embeddings(model, names, "none")
    emb = emb.to(model.device)
    queries = [f"a photo of a {l}" for l in labels[1:]]

    m_dense = IoUMeter(K, ignore)
    m_prune = IoUMeter(K, ignore)
    for ip, gp, loader in samples[offset:offset + limit]:
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
        supported[0] = True
        for li in det["labels"].tolist():
            supported[li + 1] = True
        allowed = torch.from_numpy(supported).to(logits.device)
        lg = logits.clone()
        lg[~allowed] = -1e4
        prune = lg.argmax(0).cpu().numpy().astype(np.int64)

        gt = loader(gp)
        m_dense.update(pred, gt)
        m_prune.update(prune, gt)
    del model
    torch.cuda.empty_cache()
    return m_dense.miou()[0] * 100, m_prune.miou()[0] * 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    res = {"prereg": "prereg_w12_k4_redteam.md"}
    cells = []
    # ANS red-team: heldout split (offset=100, limit=200)
    for v in ("sclip", "clearclip"):
        for vf in ("perturbed_vocabs/voc21_ans_clearclip.json",
                   "perturbed_vocabs/voc21_plain.json"):
            cells.append((v, "voc21", vf, 100, 200))
    # ADE-150 scale cells (test-300)
    for v in ("sclip", "naclip"):
        cells.append((v, "ade150", "perturbed_vocabs/ade150_plain.json",
                      0, 300))
    for v, ds, vf, off, lim in cells:
        base, pruned = cell(v, ds, vf, proc, owl, off, lim)
        key = f"{v}/{ds}/{vf.split('/')[-1]}/off{off}"
        res[key] = {"dense": base, "pruned": pruned}
        print(key, round(base, 2), "->", round(pruned, 2), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
