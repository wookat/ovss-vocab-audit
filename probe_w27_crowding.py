"""W27 (prereg_w27_crowding_expand.md): crowding signal, independent-point
expansion under one matched extraction protocol.

18 points: {sclip, naclip, clearclip} x {voc21/plain, cocoobj/plain,
ctx60/plain, ade150/plain, voc21/syn100_s0, cocoobj/syn100_s0}.
Per point: S1 = mean max raw cosine sim (first 50 unlabeled images) and
oracle gain = mIoU(vocab+VABS64) - mIoU(vocab), test-300 standard meter.
"""
import json
import os
import numpy as np
import torch
from PIL import Image
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, evaluate
import data

A = os.path.dirname(os.path.abspath(__file__))
V = f"{A}/perturbed_vocabs"
N_IMGS = 50
LIMIT = 300

CELLS = [
    ("voc21", "voc21_plain.json", "voc21_plain_vabs64.json"),
    ("cocoobj", "cocoobj_plain.json", "cocoobj_plain_vabs.json"),
    ("ctx60", "ctx60_plain.json", "ctx60_plain_vabs64.json"),
    ("ade150", "ade150_plain.json", "ade150_plain_vabs64.json"),
    ("voc21", "voc21_syn100_s0.json", "voc21_syn100_s0_vabs64.json"),
    ("cocoobj", "cocoobj_syn100_s0.json", "cocoobj_syn100_s0_vabs64.json"),
]
LOADERS = {"voc21": data.voc21, "cocoobj": data.cocoobj, "ctx60": data.ctx60,
           "ade150": data.ade150}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def crowding(model, samples, names):
    text_emb, _ = class_embeddings(model, names)
    text_emb = text_emb.to(model.device)
    s = n = 0.0
    for img_path, _, _ in samples[:N_IMGS]:
        img = Image.open(img_path).convert("RGB")
        img_r, _ = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, text_emb)
        s += float(sims.max(0).values.mean())
        n += 1
    return s / n


def main():
    out = {}
    for m in ("sclip", "naclip", "clearclip"):
        model = DenseCLIP(m, device="cuda")
        for ds, vf, vabsf in CELLS:
            key = f"{m}/{vf}"
            names = json.load(open(f"{V}/{vf}"))
            vnames = json.load(open(f"{V}/{vabsf}"))
            samples = LOADERS[ds]()[0]
            s1 = crowding(model, samples, names)
            base = evaluate(model, samples, names, limit=LIMIT)[0]
            vabs = evaluate(model, samples, vnames, limit=LIMIT)[0]
            out[key] = {"S1": s1, "base": base, "vabs": vabs,
                        "gain": vabs - base, "dataset": ds}
            print(f"{key}: S1={s1:.4f} base={base:.2f} vabs={vabs:.2f} "
                  f"gain={vabs-base:+.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    xs = [d["S1"] for d in out.values()]
    ys = [d["gain"] for d in out.values()]
    res = {"points": out, "rho_pooled": spearman(xs, ys)}
    print(f"H1 pooled rho = {res['rho_pooled']:.3f}")
    for m in ("sclip", "naclip", "clearclip"):
        ks = [k for k in out if k.startswith(m + "/")]
        r = spearman([out[k]["S1"] for k in ks], [out[k]["gain"] for k in ks])
        res[f"rho_{m}"] = r
        print(f"H2 {m}: rho = {r:.3f}")
    h3 = 0
    for m in ("sclip", "naclip", "clearclip"):
        for ds in ("voc21", "cocoobj"):
            p = out[f"{m}/{ds}_plain.json"]
            s = out[f"{m}/{ds}_syn100_s0.json"]
            hi, lo = (p, s) if p["S1"] > s["S1"] else (s, p)
            if hi["gain"] < lo["gain"]:
                h3 += 1
    res["h3_correct_pairs"] = h3
    print(f"H3 correct pairs = {h3}/6")
    json.dump(res, open("/media/dell/DATA/ovss/runs/w27_crowding.json", "w"),
              indent=1)
    print("wrote /media/dell/DATA/ovss/runs/w27_crowding.json")


if __name__ == "__main__":
    main()
