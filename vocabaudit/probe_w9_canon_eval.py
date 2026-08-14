"""W9-H2 dense-arm evaluation (prereg_w9_h2_canon.md)."""
import argparse
import json

import torch

import data
from clip_seg import DenseCLIP
from eval_seg import evaluate

VOCABS = ["voc21_plain", "voc21_plain_canon",
          "voc21_syn100_s0", "voc21_syn100_s0_canon",
          "voc21_freqctrl_s0", "voc21_freqctrl_s0_canon"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, _, ignore = data.DATASETS["voc21"]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"prereg": "prereg_w9_h2_canon.md"}
    for variant in ["sclip", "clearclip"]:
        model = DenseCLIP(variant, device=dev)
        for vn in VOCABS:
            vocab = json.load(open(f"perturbed_vocabs/{vn}.json"))
            miou, _, _ = evaluate(model, samples, vocab, "none", 0.5, 336,
                                  limit=300, ignore=ignore, offset=0,
                                  log_every=10000)
            res[f"{variant}/{vn}"] = miou * 100
            print(variant, vn, res[f"{variant}/{vn}"], flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        del model
        torch.cuda.empty_cache()
