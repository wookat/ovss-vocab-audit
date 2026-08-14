"""W13-L4 (prereg_w13_l4_granularity.md): OWLv2 presence recall vs label
granularity on ADE-150, correlated with per-class J5 pruning gain."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr
from transformers import Owlv2ForObjectDetection, Owlv2Processor

import data
from clip_seg import DenseCLIP
from eval_seg import (IoUMeter, class_embeddings, resize_short, to_tensor,
                      seg_logits)

THRESH = 0.2


def partial_spearman(x, y, z):
    def resid(a, b):
        ar = np.argsort(np.argsort(a)).astype(float)
        br = np.argsort(np.argsort(b)).astype(float)
        beta = np.polyfit(br, ar, 1)
        return ar - np.polyval(beta, br)
    return spearmanr(resid(x, z), resid(y, z)).statistic


@torch.no_grad()
def run(out_path, limit=300):
    samples, _, ignore = data.DATASETS["ade150"]()
    names = json.load(open("perturbed_vocabs/ade150_plain.json"))
    labels = [n.split(",")[0].strip() for n in names]
    K = len(names)
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    model = DenseCLIP("sclip", device="cuda")
    emb, qidx = class_embeddings(model, names, "none")
    emb = emb.to(model.device)
    queries = [f"a photo of a {l}" for l in labels[1:]]

    m_dense = IoUMeter(K, ignore)
    m_prune = IoUMeter(K, ignore)
    n_present = np.zeros(K)
    n_detected = np.zeros(K)
    pix_freq = np.zeros(K)

    for ip, gp, loader in samples[:limit]:
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        lg = seg_logits(model, t, emb, 224, 112)
        lg = F.interpolate(lg.unsqueeze(0), size=(h0, w0),
                           mode="bilinear", align_corners=False)[0]
        m_dense.update(lg.argmax(0).cpu().numpy().astype(np.int64), gt)

        inputs = proc(text=[queries], images=img, padding="max_length",
                      return_tensors="pt").to("cuda")
        out = owl(**inputs)
        det = proc.post_process_object_detection(
            out, threshold=THRESH, target_sizes=torch.tensor([(h0, w0)]))[0]
        supported = np.zeros(K, dtype=bool)
        supported[0] = True
        for li in det["labels"].tolist():
            supported[li + 1] = True
        l2 = lg.clone()
        l2[torch.from_numpy(~supported).to(l2.device)] = -1e4
        m_prune.update(l2.argmax(0).cpu().numpy().astype(np.int64), gt)

        present = np.unique(gt[gt != ignore])
        for c in present:
            if 0 < c < K:
                n_present[c] += 1
                if supported[c]:
                    n_detected[c] += 1
                pix_freq[c] += float((gt == c).sum())

    valid = n_present >= 5
    valid[0] = False
    recall = n_detected / np.maximum(n_present, 1)
    iou_d = m_dense.inter / np.maximum(m_dense.union, 1)
    iou_p = m_prune.inter / np.maximum(m_prune.union, 1)
    gain = (iou_p - iou_d) * 100

    depth = np.full(K, np.nan)
    for c in range(1, K):
        ss = wn.synsets(labels[c].replace(" ", "_"), pos=wn.NOUN)
        if ss:
            depth[c] = min(s.min_depth() for s in ss)
    has_syn = ~np.isnan(depth)
    m1 = valid & has_syn
    logf = np.log(np.maximum(pix_freq, 1))

    res = {
        "prereg": "prereg_w13_l4_granularity.md",
        "dense_miou": m_dense.miou()[0] * 100,
        "prune_miou": m_prune.miou()[0] * 100,
        "n_classes_valid": int(valid.sum()),
        "n_no_synset": int((valid & ~has_syn).sum()),
        "rho_depth_recall": spearmanr(depth[m1], recall[m1]).statistic,
        "rho_recall_gain": spearmanr(recall[valid], gain[valid]).statistic,
        "partial_depth_recall_ctrl_freq":
            partial_spearman(depth[m1], recall[m1], logf[m1]),
        "mean_recall": float(recall[valid].mean()),
    }
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.out)
