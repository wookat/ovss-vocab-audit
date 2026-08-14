"""W4c E2: transplant LPOSS-style propagation onto the other six methods.

For each base method, evaluate {plain, syn100_s0} with and without post-hoc
DINO-affinity propagation (alpha=0.9, 10 iters, k=32; frozen in
prereg_w4c_propsmooth.md). GO if mean synonym-drop reduction >= 25% without
losing > 1.0 plain mIoU on any method.
"""
import argparse
import json
import os
import torch
from clip_seg import DenseCLIP
from eval_seg import evaluate
from newgen_seg import PropVariant, PropSCCLIP, PropProxy, SCCLIP
from proxyclip_seg import ProxyCLIP
import data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    choices=["maskclip", "sclip", "clearclip", "naclip",
                             "proxyclip", "scclip"])
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, names, ignore = data.DATASETS[a.dataset]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"base": a.base, "dataset": a.dataset, "limit": a.limit, "arms": {}}
    for prop in (False, True):
        if a.base == "proxyclip":
            model = PropProxy(device=dev) if prop else ProxyCLIP(device=dev)
        elif a.base == "scclip":
            model = PropSCCLIP(device=dev) if prop else SCCLIP(device=dev)
        else:
            model = (PropVariant(a.base, device=dev) if prop
                     else DenseCLIP(a.base, device=dev))
        for vk in ("plain", "syn100_s0"):
            vocab = json.load(open(f"perturbed_vocabs/{a.dataset}_{vk}.json"))
            miou, _, miou_all = evaluate(model, samples, vocab, "none", 0.5, 336,
                                         limit=a.limit, ignore=ignore,
                                         offset=a.offset)
            key = f"{'prop' if prop else 'base'}_{vk}"
            res["arms"][key] = {"miou": miou, "miou_all": miou_all}
            print(key, round(miou * 100, 2), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
