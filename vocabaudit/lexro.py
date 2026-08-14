"""LexRO adapter definition + variant pool construction (prereg_lexro_v1.md)."""
import json, os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn


class TextAdapter(nn.Module):
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, e):
        return F.normalize(e + self.fc2(F.gelu(self.fc1(e))), dim=-1)


def eval_variant_names(vocab_dir="perturbed_vocabs"):
    """All names occurring in synonym EVAL vocab files (excluded from training)."""
    banned = set()
    for f in glob.glob(f"{vocab_dir}/voc21_syn*.json") + glob.glob(f"{vocab_dir}/coco171_syn*.json"):
        for n in json.load(open(f)):
            for s in n.split(","):
                banned.add(s.strip().lower())
    return banned


def build_variant_pool(model, class_names, banned, lo=0.70, hi=0.95):
    """Per class: [canonical] + WordNet synonyms with CLIP cos in [lo,hi], minus banned."""
    pool = []
    for n in class_names:
        base = n.split(",")[0].strip().lower()
        variants = [base]
        if base not in ("background", "unknown", "other"):
            cands = []
            for syn in wn.synsets(base.replace(" ", "_"), pos=wn.NOUN):
                for l in syn.lemmas():
                    w = l.name().replace("_", " ").lower()
                    if w != base and w not in cands and w not in banned:
                        cands.append(w)
            if cands:
                eb = model.encode_text_raw([f"a photo of a {base}."])
                ec = model.encode_text_raw([f"a photo of a {c}." for c in cands])
                cos = (ec @ eb[0]).tolist()
                variants += [c for c, s in zip(cands, cos) if lo <= s <= hi]
        pool.append(variants)
    return pool
