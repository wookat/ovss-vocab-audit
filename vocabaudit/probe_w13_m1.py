"""W13-M1 (prereg_w13_m1_ledger.md): pruning profit/loss pixel ledger."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

import data
from clip_seg import DenseCLIP
from eval_seg import IoUMeter, class_embeddings, resize_short, to_tensor, \
    seg_logits

THRESH = 0.2


@torch.no_grad()
def cell(variant, ds, proc, owl, limit=300):
    samples, _, ignore = data.DATASETS[ds]()
    names = json.load(open(f"perturbed_vocabs/{ds}_plain.json"))
    labels = [n.split(",")[0].strip() for n in names]
    K = len(names)
    has_bg = ds != "ade150"
    model = DenseCLIP(variant, device="cuda")
    emb, _ = class_embeddings(model, names, "none")
    emb = emb.to(model.device)
    queries = [f"a photo of a {l}" for l in (labels[1:] if has_bg else labels)]

    led = dict(A=0, B=0, C=0, changed=0, total=0,
               n_pres=0, n_pres_pruned=0)
    iou_pruned = IoUMeter(K, ignore)   # dense IoU of mispruned present cls
    iou_kept = IoUMeter(K, ignore)     # dense IoU of kept present cls
    area_pruned, area_kept = 0, 0
    for ip, gp, loader in samples[:limit]:
        gt = loader(gp)
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
        off = 1 if has_bg else 0
        if has_bg:
            supported[0] = True
        for li in det["labels"].tolist():
            supported[li + off] = True
        allowed = torch.from_numpy(supported).to(logits.device)
        lg = logits.clone()
        lg[~allowed] = -1e4
        prune = lg.argmax(0).cpu().numpy().astype(np.int64)

        m = gt != ignore
        p0, p1, g = pred[m], prune[m], gt[m]
        ch = p0 != p1
        led["changed"] += int(ch.sum())
        led["total"] += int(m.sum())
        led["A"] += int(((p0 == g) & ch).sum())
        led["B"] += int(((p0 != g) & (p1 == g)).sum())
        led["C"] += int(((p0 != g) & ch & (p1 != g)).sum())

        pres = np.unique(g)
        pres = pres[pres < K]
        for c in pres:
            if not has_bg or c != 0:
                led["n_pres"] += 1
                cm = g == c
                dense_pred_c = (p0 == c)
                # per-class dense IoU on this image
                inter = int((cm & dense_pred_c).sum())
                union = int((cm | dense_pred_c).sum())
                tgt = iou_kept if supported[c] else iou_pruned
                tgt.inter[c] += inter
                tgt.union[c] += union
                tgt.seen[c] += 1
                if supported[c]:
                    area_kept += int(cm.sum())
                else:
                    led["n_pres_pruned"] += 1
                    area_pruned += int(cm.sum())
    del model
    torch.cuda.empty_cache()
    led["dense_iou_mispruned"] = iou_pruned.miou()[0] * 100
    led["dense_iou_kept"] = iou_kept.miou()[0] * 100
    led["mean_area_mispruned"] = (area_pruned / max(led["n_pres_pruned"], 1))
    led["mean_area_kept"] = (area_kept /
                             max(led["n_pres"] - led["n_pres_pruned"], 1))
    return led


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    res = {"prereg": "prereg_w13_m1_ledger.md"}
    for v in ("sclip", "naclip"):
        for ds in ("voc21", "ade150"):
            led = cell(v, ds, proc, owl)
            res[f"{v}/{ds}"] = led
            print(v, ds, {k: (round(x, 3) if isinstance(x, float) else x)
                          for k, x in led.items()}, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
