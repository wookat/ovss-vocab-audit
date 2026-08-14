"""W11-J3 (prereg_w11_j3_translation.md): SCLIP per-class IoU for k=3
translations of a language; best-of-3 = per-class max."""
import argparse
import json

import numpy as np

import data
from clip_seg import DenseCLIP
from probe_w10_es_bestofk import per_class_iou

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, _, ignore = data.DATASETS["voc21"]()
    model = DenseCLIP("sclip", device="cuda")
    res = {"prereg": "prereg_w11_j3_translation.md", "lang": a.lang}
    ious = []
    for tag in [a.lang, f"{a.lang}_alt1", f"{a.lang}_alt2"]:
        names = json.load(open(f"perturbed_vocabs/voc21_{tag}.json"))
        iou, present = per_class_iou(model, samples, names, 300, ignore)
        res[tag] = float(iou[present].mean() * 100)
        ious.append(iou)
        print(tag, res[tag], flush=True)
    best = np.maximum.reduce(ious)
    res["best_of_3"] = float(best[present].mean() * 100)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
