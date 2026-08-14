"""W4f addendum: ANS transfer matrix. Evaluate each method's searched
vocabulary on the other method (held-out test-200), to separate per-method
search adaptation from genuine shared fragility."""
import argparse
import json
import torch

import data
from clip_seg import DenseCLIP
from eval_seg import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, names, ignore = data.DATASETS["voc21"]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocabs = {m: json.load(open(f"/media/dell/DATA/ovss/runs/w4f_ans_{m}.json"))["ans_vocab"]
              for m in ("clearclip", "lposs")}
    res = {}
    for m in ("clearclip", "lposs"):
        if m == "lposs":
            from newgen_seg import LPOSS
            model = LPOSS(device=dev)
        else:
            model = DenseCLIP(m, device=dev)
        for src, vocab in vocabs.items():
            miou, _, _ = evaluate(model, samples, vocab, "none", 0.5, 336,
                                  limit=200, ignore=ignore, offset=100,
                                  log_every=10000)
            res[f"{src}_vocab_on_{m}"] = miou * 100
            print(src, "->", m, round(miou * 100, 2), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
