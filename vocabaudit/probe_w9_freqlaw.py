"""W9 (H1) go/no-go: name-level frequency law (prereg_w9_h1_freqlaw.md).

Per-class IoU deltas from archived confusion matrices; frequency via
wordfreq zipf (frozen primary estimator). Runs locally on pulled npz files.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from wordfreq import zipf_frequency

VOCDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "perturbed_vocabs")


def zipf(name):
    toks = name.split()
    return min(zipf_frequency(t, "en") for t in toks)


def per_class_iou(conf):
    kg = conf.shape[0]
    inter = np.diag(conf[:, :kg]).astype(float)
    union = conf.sum(1)[:kg] + conf[:, :kg].sum(0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    return iou


def load(path):
    d = np.load(path)
    return per_class_iou(d["conf"])


def names(fn):
    return [n.split(",")[0].strip()
            for n in json.load(open(os.path.join(VOCDIR, fn)))]


def token_count(name):
    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-16")
    return int((tok([name]) != 0).sum()) - 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/tmp/w9npz")
    ap.add_argument("--ans-run", default="/tmp/w4f_ans_clearclip.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    obs = []  # (model, dataset, class_idx, plain_name, new_name, delta)

    def add_pair(model, ds, plain_iou, pert_iou, plain_names, pert_names):
        for c, (pn, qn) in enumerate(zip(plain_names, pert_names)):
            if pn == qn or c == 0:
                continue
            if np.isnan(plain_iou[c]) or np.isnan(pert_iou[c]):
                continue
            obs.append({"model": model, "ds": ds, "cls": c, "plain": pn,
                        "name": qn,
                        "delta": float((pert_iou[c] - plain_iou[c]) * 100)})

    dense = ["maskclip", "sclip", "clearclip", "naclip",
             "proxyclip", "lposs", "scclip"]
    for ds, tag in [("voc21", "voc21"), ("coco171", "coco171")]:
        pl = names(f"{tag}_plain.json")
        sy = names(f"{tag}_syn100_s0.json")
        for m in dense:
            p = f"{a.npz_dir}/w5e_{m}_{tag}_plain.npz"
            q = f"{a.npz_dir}/w5e_{m}_{tag}_syn100_s0.npz"
            if os.path.exists(p) and os.path.exists(q):
                add_pair(m, ds, load(p), load(q), pl, sy)

    pl = names("voc21_plain.json")
    sy = names("voc21_syn100_s0.json")
    add_pair("owlv2", "voc21",
             load(f"{a.npz_dir}/w6f2_owl_plain.json.npz"),
             load(f"{a.npz_dir}/w6f2_owl_syn100_s0.json.npz"), pl, sy)
    add_pair("gdino", "voc21",
             load(f"{a.npz_dir}/w6f2_gdino_plain.json.npz"),
             load(f"{a.npz_dir}/w6f2_gdino_syn100_s0.json.npz"), pl, sy)

    plain_h = load(f"{a.npz_dir}/w6f2_owl_plain_h200.json.npz")
    ans_vocab = [n.split(",")[0].strip()
                 for n in json.load(open(a.ans_run))["ans_vocab"]]
    add_pair("owlv2", "voc21-h200", plain_h,
             load(f"{a.npz_dir}/w6f2_owl_ans.json.npz"), pl, ans_vocab)
    for s in range(3):
        fc = names(f"voc21_freqctrl_s{s}.json")
        add_pair("owlv2", "voc21-h200", plain_h,
                 load(f"{a.npz_dir}/w7a_freqctrl_owl_s{s}.json.npz"), pl, fc)

    for o in obs:
        o["zipf"] = zipf(o["name"])
        o["zipf_plain"] = zipf(o["plain"])
        o["ntok"] = token_count(o["name"])

    models = sorted(set(o["model"] for o in obs))
    rows = {}
    for m in models:
        sub = [o for o in obs if o["model"] == m]
        d = [-o["delta"] for o in sub]  # drop = plain - perturbed
        f = [-o["zipf"] for o in sub]  # rarity
        t = [o["ntok"] for o in sub]
        rho, _ = spearmanr(d, f)
        # partial Spearman controlling token count via rank regression
        def resid(y, x):
            from scipy.stats import rankdata
            ry, rx = rankdata(y), rankdata(x)
            b = np.polyfit(rx, ry, 1)
            return ry - np.polyval(b, rx)
        pr, _ = spearmanr(resid(d, t), resid(f, t))
        rows[m] = {"n": len(sub), "spearman_damage_vs_rarity": float(rho),
                   "partial_ctrl_ntok": float(pr)}

    # within-concept sign consistency: classes with >=3 name observations
    from collections import defaultdict
    per_cls = defaultdict(list)
    for o in obs:
        per_cls[(o["ds"], o["cls"])].append(o)
    consist = []
    for k, v in per_cls.items():
        if len(v) < 3:
            continue
        rho, _ = spearmanr([-o["delta"] for o in v], [-o["zipf"] for o in v])
        if not np.isnan(rho):
            consist.append(rho)
    med = float(np.median([r["spearman_damage_vs_rarity"]
                           for r in rows.values()]))
    out = {"prereg": "prereg_w9_h1_freqlaw.md", "n_obs": len(obs),
           "per_model": rows, "median_spearman": med,
           "within_concept_n": len(consist),
           "within_concept_frac_positive_rho": float(
               np.mean([r > 0 for r in consist])) if consist else None,
           "within_concept_median_rho": float(np.median(consist))
           if consist else None}
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(obs, open(a.out + ".obs.json", "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
