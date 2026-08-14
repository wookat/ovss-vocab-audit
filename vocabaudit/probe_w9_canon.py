"""W9-H2: name canonicalization (prereg_w9_h2_canon.md). Generates
canonicalized versions of vocab files; frozen rule: highest-zipf alias from
WordNet first-sense lemmas + head noun, guarded by CLIP text cosine >= 0.80,
tie-break shorter BPE."""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import open_clip
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

VOCDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "perturbed_vocabs")


def zipf(name):
    return min(zipf_frequency(t, "en") for t in name.split())


def candidates(name):
    cands = {name}
    parts = name.split()
    if len(parts) > 1:
        cands.add(parts[-1])
    ss = wn.synsets(name.replace(" ", "_"), pos=wn.NOUN)
    if ss:
        for lemma in ss[0].lemmas():
            cands.add(lemma.name().replace("_", " ").lower())
    return sorted(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabs", nargs="+", required=True)
    ap.add_argument("--suffix", default="_canon")
    a = ap.parse_args()

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-16")
    model.eval()

    def embed(words):
        with torch.no_grad():
            e = model.encode_text(tok([f"a photo of a {w}." for w in words]))
        return F.normalize(e.float(), dim=-1)

    def ntok(w):
        return int((tok([w]) != 0).sum()) - 2

    for vf in a.vocabs:
        names = [n.split(",")[0].strip()
                 for n in json.load(open(os.path.join(VOCDIR, vf)))]
        out = []
        for i, name in enumerate(names):
            if i == 0:
                out.append(name)
                continue
            cands = candidates(name)
            E = embed([name] + cands)
            cos = (E[1:] @ E[0]).tolist()
            ok = [(w, c) for w, c in zip(cands, cos) if c >= 0.80]
            if not ok:
                out.append(name)
                continue
            pick = sorted(ok, key=lambda wc: (-zipf(wc[0]), ntok(wc[0])))[0][0]
            out.append(pick)
        of = vf.replace(".json", f"{a.suffix}.json")
        json.dump(out, open(os.path.join(VOCDIR, of), "w"))
        changed = sum(1 for x, y in zip(names, out) if x != y)
        print(of, "changed:", changed, flush=True)


if __name__ == "__main__":
    main()
