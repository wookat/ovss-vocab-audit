"""W11-J1 alias-diversity measurement (prereg_w11_j1_alias.md). Local, no GPU."""
import json
import math
import re

import ijson
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POOL = json.load(open("/tmp/w11_alias_pool.json"))
CANON = list(POOL.keys())
GRIT_PARQUET = ("/home/ubuntu/.cache/huggingface/hub/datasets--zzliang--GRIT/"
                "snapshots/4696bb040fe6d961a4ae3bac4e9d0c51d10d8f72/"
                "grit-20m/coyo_0_snappy.parquet")
GOLDG = ["/home/ubuntu/goldg/OpenSource/final_mixed_train.json",
         "/home/ubuntu/goldg/OpenSource/final_flickr_mergedGT_train.json"]

def alias_sets():
    out = {}
    for c, pool in POOL.items():
        aliases = [c.replace("diningtable", "dining table")
                    .replace("tvmonitor", "tv monitor")] + pool
        out[c] = {a: re.compile(r"\b" + re.escape(a) + r"\b") for a in aliases}
    return out

AL = alias_sets()

def count_in_texts(texts):
    counts = {c: {a: 0 for a in AL[c]} for c in CANON}
    for t in texts:
        tl = t.lower()
        for c in CANON:
            for a, rx in AL[c].items():
                if rx.search(tl):
                    counts[c][a] += 1
    return counts

def merge(*cs):
    out = {c: {a: 0 for a in AL[c]} for c in CANON}
    for cnt in cs:
        for c in CANON:
            for a in AL[c]:
                out[c][a] += cnt[c][a]
    return out

def entropy(cnt):
    ent = {}
    for c in CANON:
        v = np.array([x for x in cnt[c].values() if x > 0], dtype=float)
        ent[c] = 0.0 if v.size <= 1 else float(
            -(v / v.sum() * np.log2(v / v.sum())).sum())
    return ent

def goldg_texts():
    for f in GOLDG:
        with open(f, "rb") as fh:
            for img in ijson.items(fh, "images.item"):
                yield img.get("caption", "")

print("counting GoldG ...", flush=True)
gold = count_in_texts(goldg_texts())
print("counting GRIT sample (1M captions) ...", flush=True)
caps = pd.read_parquet(GRIT_PARQUET, columns=["caption"])["caption"]
grit = count_in_texts(caps)
v3det = count_in_texts(open("/tmp/v3det_cats.txt").read().lower().splitlines())
o365_names = re.findall(r"'([^']+)'", open("/tmp/o365.py").read())
o365 = count_in_texts([n.lower() for n in o365_names])

tiers = {"T1_goldg": gold, "T2_goldg_grit": merge(gold, grit),
         "T3_goldg_grit_v3det": merge(gold, grit, v3det)}
res = {"prereg": "prereg_w11_j1_alias.md",
       "corpus_counts": {"goldg": gold, "grit": grit, "v3det": v3det,
                         "o365": o365}}
for t, cnt in tiers.items():
    e = entropy(cnt)
    res[t] = {"per_class_entropy": e,
              "mean_entropy": float(np.mean(list(e.values())))}
    print(t, round(res[t]["mean_entropy"], 3))

# class-level: damage reduction T1->T2 vs GRIT alias entropy
def pciou(f):
    d = np.load(f)
    c = d["conf"].astype(float)
    kg = int(d["kg"])
    inter = np.diag(c[:, :kg])
    union = c.sum(1)[:kg] + c[:, :kg].sum(0) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)

def drop(tier):
    return (pciou(f"/tmp/w10_{tier}_plain.json.npz")
            - pciou(f"/tmp/w10_{tier}_syn100_s0.json.npz"))

d1, d2 = drop("t1"), drop("t2")
red = (d1 - d2)[1:]  # damage reduction per fg class
ge = np.array([entropy(grit)[c] for c in CANON])
ok = ~np.isnan(red)
rho, p = spearmanr(red[ok], ge[ok])
res["class_level"] = {"spearman_reduction_vs_grit_entropy": float(rho),
                      "p": float(p), "n": int(ok.sum())}
print("class-level rho", round(rho, 3), "p", round(p, 3), "n", int(ok.sum()))
json.dump(res, open("/tmp/w11_j1_alias.json", "w"), indent=1)
