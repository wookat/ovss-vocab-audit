"""W7a evaluation (prereg_w7a_freqctrl.md): token-matched control vocabs on
the ClearCLIP pixel baseline, VOC held-out-200 (offset 100), matching the
ANS held-out protocol."""
import argparse
import json
import torch

import data
from clip_seg import DenseCLIP
from eval_seg import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, _, ignore = data.DATASETS[a.dataset]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DenseCLIP("clearclip", device=dev)

    res = {"prereg": "prereg_w7a_freqctrl.md", "variant": "clearclip"}
    for tag in ["freqctrl_s0", "freqctrl_s1", "freqctrl_s2"]:
        vocab = json.load(open(f"perturbed_vocabs/{a.dataset}_{tag}.json"))
        miou, _, _ = evaluate(model, samples, vocab, "none", 0.5, 336,
                              limit=200, ignore=ignore, offset=100,
                              log_every=10000)
        res[tag] = miou * 100
        print(tag, res[tag], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
