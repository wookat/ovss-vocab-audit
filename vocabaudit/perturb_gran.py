"""Axis 2: granularity shift via frozen WordNet rule (no per-class hand-picking).

coarse: replace each class name by the lemma of its first-sense direct hypernym.
Classes whose name has no noun synset keep the original name (logged).
Labels unchanged; per card2 the coarser name still scores against the original GT class.
Also emits a merge map (classes sharing an identical coarse name) for the merged-credit
scoring variant reported in the appendix.
"""
import json, os
from nltk.corpus import wordnet as wn
import openseg_classes as oc

OUT = "perturbed_vocabs"
DATASETS = {
    "voc21": [c["name"].split(",")[0].strip() for c in oc.PASCAL_VOC_21_CATEGORIES],
    "ade150": [c["name"].split(",")[0].strip() for c in oc.ADE20K_150_CATEGORIES],
}


EXCLUDE = {"background", "unknown", "other"}  # non-semantic classes never perturbed (all axes)


def coarse_name(name):
    if name in EXCLUDE:
        return None
    syns = wn.synsets(name.replace(" ", "_"), pos=wn.NOUN)
    if not syns:
        return None
    hyper = syns[0].hypernyms()
    if not hyper:
        return None
    return hyper[0].lemmas()[0].name().replace("_", " ").lower()


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for ds, names in DATASETS.items():
        vocab, mapping, kept = [], {}, []
        for n in names:
            c = coarse_name(n)
            if c is None:
                vocab.append(n); kept.append(n)
            else:
                vocab.append(c); mapping[n] = c
        json.dump(vocab, open(f"{OUT}/{ds}_gran_coarse.json", "w"), indent=1)
        merge = {}
        for i, v in enumerate(vocab):
            merge.setdefault(v, []).append(i)
        manifest[ds] = dict(mapping=mapping, unchanged=kept,
                            merged_groups={k: v for k, v in merge.items() if len(v) > 1})
        print(ds, len(mapping), "coarsened,", len(kept), "kept,",
              sum(len(v) > 1 for v in merge.values()), "merge groups")
    json.dump(manifest, open(f"{OUT}/gran_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
