"""W8: Spanish naming axis across the dense method matrix (prereg_w8_es_axis.md)."""
import argparse
import json

import torch

import data
from clip_seg import DenseCLIP
from eval_seg import evaluate


def load(variant, dev):
    if variant == "lposs":
        from newgen_seg import LPOSS
        return LPOSS(device=dev)
    if variant == "scclip":
        from newgen_seg import SCCLIP
        return SCCLIP(device=dev)
    if variant == "proxyclip":
        from proxyclip_seg import ProxyCLIP
        return ProxyCLIP(device=dev)
    return DenseCLIP(variant, device=dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, _, ignore = data.DATASETS["voc21"]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open("perturbed_vocabs/voc21_es.json"))
    res = {"prereg": "prereg_w8_es_axis.md"}
    for v in ["maskclip", "sclip", "clearclip", "naclip",
              "proxyclip", "lposs", "scclip"]:
        model = load(v, dev)
        miou, _, _ = evaluate(model, samples, vocab, "none", 0.5, 336,
                              limit=300, ignore=ignore, offset=0,
                              log_every=10000)
        res[v] = miou * 100
        print(v, res[v], flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1)
