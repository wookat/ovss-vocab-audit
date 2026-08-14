"""W6-F3 (prereg_w6f3_law.md, frozen): text-only vulnerability-prediction law.

Computes the frozen geometry signal z(g1)+z(g2) per vocabulary and Spearman
against archived per-method mIoU across the vocabulary pool. No new
segmentation runs.
"""
import argparse
import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from clip_seg import DenseCLIP
from eval_seg import class_embeddings

POOLS = {
    "voc21": ("w4a_voc21full_{m}_{v}.json",
              ["plain", "official", "syn100_s0", "syn50_s0", "syn50_s1", "syn50_s2"],
              "voc21_{v}.json"),
    "cocoobj": ("w3d_cocoobj_{m}_{v}.json",
                ["plain", "syn100_s0", "syn50_s0", "syn50_s1", "syn50_s2"],
                "cocoobj_{v}.json"),
    "ctx60": ("w3d_ctx60_{m}_{v}.json",
              ["plain", "syn100_s0", "syn50_s0", "syn50_s1", "syn50_s2"],
              "ctx60_{v}.json"),
}
METHODS = ["maskclip", "sclip", "clearclip", "naclip", "proxyclip", "lposs", "scclip"]


def class_emb_pooled(model, names):
    T, qi = class_embeddings(model, names)
    K = int(qi.max()) + 1
    out = []
    for c in range(K):
        out.append(F.normalize(T[qi == c].mean(0), dim=-1))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/media/dell/DATA/ovss/runs")
    ap.add_argument("--vocab-dir", default="perturbed_vocabs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = DenseCLIP("clearclip")  # text encoder only is used
    res = {"prereg": "prereg_w6f3_law.md", "datasets": {}}
    for ds, (pat, vocabs, vpat) in POOLS.items():
        plain_names = json.load(open(os.path.join(a.vocab_dir, vpat.format(v="plain"))))
        E_plain = class_emb_pooled(model, plain_names)
        g1s, g2s = [], []
        for v in vocabs:
            names = json.load(open(os.path.join(a.vocab_dir, vpat.format(v=v))))
            E = class_emb_pooled(model, names)
            S = E @ E.T
            S.fill_diagonal_(-1)
            g1 = float((1 - S.max(1).values).mean())
            g2 = float((E * E_plain).sum(-1).mean())
            g1s.append(g1)
            g2s.append(g2)
        g1s, g2s = np.array(g1s), np.array(g2s)
        z = lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        score = z(g1s) + z(g2s)
        dsres = {"vocabs": vocabs, "g1": g1s.tolist(), "g2": g2s.tolist(),
                 "score": score.tolist(), "methods": {}}
        rhos = []
        for m in METHODS:
            mious = []
            ok = True
            for v in vocabs:
                fp = os.path.join(a.runs_dir, pat.format(m=m, v=v))
                if not os.path.exists(fp):
                    ok = False
                    break
                mious.append(json.load(open(fp))["miou"] * 100)
            if not ok:
                dsres["methods"][m] = None
                continue
            rho = float(spearmanr(score, mious).statistic)
            dsres["methods"][m] = {"miou": mious, "spearman": rho}
            rhos.append(rho)
        dsres["median_spearman"] = float(np.median(rhos)) if rhos else None
        res["datasets"][ds] = dsres
        print(ds, "median spearman:", dsres["median_spearman"],
              {m: (r["spearman"] if r else None) for m, r in dsres["methods"].items()},
              flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
