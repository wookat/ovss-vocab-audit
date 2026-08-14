"""W12-K1a (prereg_w12_k1_mech.md): absent-class leakage accounting of
synonym damage + per-class correlation with J5 pruning gain (SCLIP/VOC)."""
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
def run(out_path, limit=300):
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    syn_names = json.load(open("perturbed_vocabs/voc21_syn100_s0.json"))
    K = len(plain_names)
    labels_syn = [n.split(",")[0].strip() for n in syn_names]

    model = DenseCLIP("sclip", device="cuda")
    emb_p, _ = class_embeddings(model, plain_names, "none")
    emb_s, _ = class_embeddings(model, syn_names, "none")
    emb_p, emb_s = emb_p.to(model.device), emb_s.to(model.device)

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    queries = [f"a photo of a {l}" for l in labels_syn[1:]]

    leak = {"present": 0, "absent": 0, "background": 0}
    # per-class absent-leak fraction (of the class's leaked pixels)
    cls_leak_abs = np.zeros(K)
    cls_leak_tot = np.zeros(K)
    m_syn = IoUMeter(K, ignore)
    m_syn_pr = IoUMeter(K, ignore)
    for i, (ip, gp, loader) in enumerate(samples[:limit]):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        lg_p = seg_logits(model, t, emb_p, 224, 112)
        lg_s = seg_logits(model, t, emb_s, 224, 112)
        lg_p = F.interpolate(lg_p.unsqueeze(0), size=(h0, w0),
                             mode="bilinear", align_corners=False)[0]
        lg_s = F.interpolate(lg_s.unsqueeze(0), size=(h0, w0),
                             mode="bilinear", align_corners=False)[0]
        pp = lg_p.argmax(0).cpu().numpy().astype(np.int64)
        ps = lg_s.argmax(0).cpu().numpy().astype(np.int64)

        present = set(int(c) for c in np.unique(gt) if 0 < c < K)
        # GT-foreground pixels correct under plain but changed under syn
        mask = (gt > 0) & (gt < K) & (pp == gt) & (ps != gt)
        dest = ps[mask]
        src = gt[mask]
        for d, s in zip(dest.ravel(), src.ravel()):
            cls_leak_tot[s] += 1
            if d == 0:
                leak["background"] += 1
            elif int(d) in present:
                leak["present"] += 1
            else:
                leak["absent"] += 1
                cls_leak_abs[s] += 1

        inputs = proc(text=[queries], images=img, padding="max_length",
                      return_tensors="pt").to("cuda")
        det = proc.post_process_object_detection(
            owl(**inputs), threshold=THRESH,
            target_sizes=torch.tensor([(h0, w0)]))[0]
        supported = np.zeros(K, dtype=bool)
        supported[0] = True
        for li in det["labels"].tolist():
            supported[li + 1] = True
        allowed = torch.from_numpy(supported).to(lg_s.device)
        lgc = lg_s.clone()
        lgc[~allowed] = -1e4
        pr = lgc.argmax(0).cpu().numpy().astype(np.int64)

        m_syn.update(ps, gt)
        m_syn_pr.update(pr, gt)
        if (i + 1) % 50 == 0:
            tot = sum(leak.values())
            print(f"[{i+1}] absent_share="
                  f"{leak['absent']/max(tot,1):.3f}", flush=True)

    iou_s = m_syn.inter / np.maximum(m_syn.union, 1)
    iou_pr = m_syn_pr.inter / np.maximum(m_syn_pr.union, 1)
    gains = (iou_pr - iou_s) * 100
    frac = np.where(cls_leak_tot > 0, cls_leak_abs / np.maximum(cls_leak_tot, 1),
                    np.nan)
    valid = ~np.isnan(frac[1:]) & ~np.isnan(gains[1:])
    from scipy.stats import spearmanr
    rho, p = spearmanr(frac[1:][valid], gains[1:][valid])

    tot = sum(leak.values())
    res = {"prereg": "prereg_w12_k1_mech.md", "leak_counts": leak,
           "absent_share": leak["absent"] / tot,
           "present_share": leak["present"] / tot,
           "background_share": leak["background"] / tot,
           "class_spearman": float(rho), "p": float(p), "n": int(valid.sum()),
           "syn_miou": m_syn.miou()[0] * 100,
           "syn_prune_miou": m_syn_pr.miou()[0] * 100,
           "per_class_absent_frac": [None if np.isnan(x) else float(x)
                                     for x in frac],
           "per_class_prune_gain": [float(x) for x in gains]}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("per_class")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.out)
