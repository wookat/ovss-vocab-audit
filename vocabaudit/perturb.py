"""Preregistered vocabulary perturbation generator (axes 1 & 3). Frozen before any mIoU run.

Axis 1 (synonym substitution): WordNet synset lemmas of the class name (first sense match),
filtered by CLIP text cosine in [0.70, 0.95] to the original name; replace {25,50,100}% of
classes; seeds {0,1,2}. Classes without a valid synonym keep the original name (logged).

Axis 3 (distractor injection): pool = union of other datasets' class names, minus names whose
max CLIP cosine to any target class > 0.92 (synonym filter). Stratified near/mid/far by max
cosine to target vocab; sizes {+50, +200}; seed 0 (pool order fixed by sort).
Outputs JSON vocab files + manifest.json with all decisions.
"""
import json, os, random, itertools
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from clip_seg import DenseCLIP
import openseg_classes as oc

OUT = "perturbed_vocabs"
DATASETS = {
    "voc21": [c["name"] for c in oc.PASCAL_VOC_21_CATEGORIES],
    "coco171": [c["name"] for c in oc.COCO_STUFF_CATEGORIES],
    "ade150": [c["name"] for c in oc.ADE20K_150_CATEGORIES],
}
ALL_POOL_SOURCES = {
    "ade847": [c["name"] for c in oc.ADE20K_847_CATEGORIES],
    "pc459": [c["name"] for c in oc.PASCAL_CTX_459_CATEGORIES],
}


def emb(model, names):
    return model.encode_text_raw([f"a photo of a {n.split(',')[0]}." for n in names])


EXCLUDE = {"background", "unknown", "other"}  # non-semantic classes never perturbed


def synonyms(name):
    base = name.split(",")[0].strip().lower()
    if base in EXCLUDE:
        return []
    out = []
    for syn in wn.synsets(base.replace(" ", "_"), pos=wn.NOUN):
        for l in syn.lemmas():
            w = l.name().replace("_", " ").lower()
            if w != base and w not in out:
                out.append(w)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    model = DenseCLIP("sclip", device="cuda" if torch.cuda.is_available() else "cpu")
    manifest = {}
    for ds, names in DATASETS.items():
        base_names = [n.split(",")[0].strip() for n in names]
        E = emb(model, base_names)
        # ---- axis 1
        syn_choice, no_syn = {}, []
        for i, n in enumerate(base_names):
            cands = synonyms(n)
            if not cands:
                no_syn.append(n); continue
            ce = emb(model, cands)
            cos = (ce @ E[i]).tolist()
            valid = [(c, s) for c, s in zip(cands, cos) if 0.70 <= s <= 0.95]
            if valid:
                syn_choice[i] = sorted(valid, key=lambda t: -t[1])[0][0]
            else:
                no_syn.append(n)
        for frac, seed in itertools.product((0.25, 0.5, 1.0), (0, 1, 2)):
            rng = random.Random(seed)
            idxs = sorted(syn_choice)
            k = int(round(frac * len(idxs)))
            chosen = set(rng.sample(idxs, k))
            vocab = [syn_choice[i] if i in chosen else n for i, n in enumerate(base_names)]
            json.dump(vocab, open(f"{OUT}/{ds}_syn{int(frac*100)}_s{seed}.json", "w"), indent=1)
        # ---- axis 3
        pool = sorted(set(n.split(",")[0].strip() for src in ALL_POOL_SOURCES.values() for n in src) - set(base_names))
        PE = emb(model, pool)
        maxcos = (PE @ E.T).max(1).values
        keep = [(p, float(c)) for p, c in zip(pool, maxcos.tolist()) if c <= 0.92]
        strata = {
            "near": [p for p, c in keep if c > 0.80],
            "mid": [p for p, c in keep if 0.65 < c <= 0.80],
            "far": [p for p, c in keep if c <= 0.65],
        }
        for stratum, cand in strata.items():
            for size in (50, 200):
                sel = cand[:size]
                json.dump(base_names + sel, open(f"{OUT}/{ds}_dis_{stratum}{size}.json", "w"), indent=1)
        manifest[ds] = dict(n_classes=len(base_names), no_synonym=no_syn,
                            synonym_map={base_names[i]: s for i, s in syn_choice.items()},
                            strata_sizes={k: len(v) for k, v in strata.items()})
        print(ds, "done:", len(syn_choice), "synonyms,", {k: len(v) for k, v in strata.items()}, flush=True)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
