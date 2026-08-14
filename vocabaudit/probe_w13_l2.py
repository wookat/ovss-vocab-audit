"""W13-L2 (prereg_w13_l2_softprune.md): J5 defensive backfill —
matched-budget CLIP image-level top-k baseline + soft pruning lambdas."""
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
LAMBDAS = (0.3, 0.5, 0.7)


@torch.no_grad()
def cell(variant, ds, vocab_file, proc, owl, limit=300):
    samples, _, ignore = data.DATASETS[ds]()
    names = json.load(open(vocab_file))
    labels = [n.split(",")[0].strip() for n in names]
    K = len(names)
    model = DenseCLIP(variant, device="cuda")
    emb, _ = class_embeddings(model, names, "none")
    emb = emb.to(model.device)
    queries = [f"a photo of a {l}" for l in labels[1:]]

    arms = ["dense", "hard", "topk"] + [f"soft{l}" for l in LAMBDAS]
    meters = {a: IoUMeter(K, ignore) for a in arms}
    for ip, gp, loader in samples[:limit]:
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        logits = seg_logits(model, t, emb, 224, 112)
        logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0),
                               mode="bilinear", align_corners=False)[0]
        meters["dense"].update(
            logits.argmax(0).cpu().numpy().astype(np.int64), gt)

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

        # hard pruning
        lg = logits.clone()
        lg[~allowed] = -1e4
        meters["hard"].update(lg.argmax(0).cpu().numpy().astype(np.int64), gt)
        # soft pruning
        for lam in LAMBDAS:
            lg = logits.clone()
            # logits are cosine-scale; prob*lam at protocol scale 40
            # corresponds to adding log(lam)/40 in cosine units
            lg[~allowed] += float(np.log(lam)) / 40.0
            meters[f"soft{lam}"].update(
                lg.argmax(0).cpu().numpy().astype(np.int64), gt)
        # matched-budget CLIP image-level top-k
        k_kept = int(supported[1:].sum())
        t224 = F.interpolate(t, size=(224, 224), mode="bilinear",
                             align_corners=False)
        ie = F.normalize(model.model.encode_image(t224).float(), dim=-1)[0]
        sims = (emb[1:] @ ie)
        topk = torch.topk(sims, min(k_kept, K - 1)).indices.cpu().numpy() + 1
        sup2 = np.zeros(K, dtype=bool)
        sup2[0] = True
        sup2[topk] = True
        lg = logits.clone()
        lg[torch.from_numpy(~sup2).to(lg.device)] = -1e4
        meters["topk"].update(lg.argmax(0).cpu().numpy().astype(np.int64), gt)
    del model
    torch.cuda.empty_cache()
    return meters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    res = {"prereg": "prereg_w13_l2_softprune.md"}
    for v in ("sclip", "naclip"):
        for ds, vf in (("voc21", "perturbed_vocabs/voc21_plain.json"),
                       ("ade150", "perturbed_vocabs/ade150_plain.json")):
            meters = cell(v, ds, vf, proc, owl)
            res[f"{v}/{ds}"] = {arm: m.miou()[0] * 100
                                for arm, m in meters.items()}
            print(f"{v}/{ds}", {k: round(x, 2)
                                for k, x in res[f"{v}/{ds}"].items()},
                  flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
