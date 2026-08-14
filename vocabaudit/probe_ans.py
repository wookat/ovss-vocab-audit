"""W4f: ANS -- adversarial naming search (prereg_w4f_ans.md, frozen).

Greedy coordinate descent over per-class synonym choices (candidate pool:
plain name + up to 5 WordNet synonyms with CLIP cosine in [0.70, 0.95]),
minimizing mIoU on a 100-image search subset; final vocabulary evaluated on
the disjoint test-200 subset. Compares against the random syn100 suite member
on the same 200 images.
"""
import argparse
import json
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn

import data
from clip_seg import DenseCLIP
from eval_seg import evaluate
from perturb import EXCLUDE, synonyms


def candidates(model, name, lo=0.70, hi=0.95, k=5):
    base = name.split(",")[0].strip()
    if base.lower() in EXCLUDE:
        return [base]
    syns = synonyms(base)
    if not syns:
        return [base]
    E = model.encode_text_raw([f"a photo of a {w}." for w in [base] + syns])
    E = F.normalize(E.float(), dim=-1)
    cos = (E[1:] @ E[0]).tolist()
    good = [w for w, c in zip(syns, cos) if lo <= c <= hi][:k]
    return [base] + good


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, names, ignore = data.DATASETS[a.dataset]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.variant == "lposs":
        from newgen_seg import LPOSS
        model = LPOSS(device=dev)
    elif a.variant == "scclip":
        from newgen_seg import SCCLIP
        model = SCCLIP(device=dev)
    elif a.variant == "proxyclip":
        from proxyclip_seg import ProxyCLIP
        model = ProxyCLIP(device=dev)
    else:
        model = DenseCLIP(a.variant, device=dev)

    plain = [n.split(",")[0].strip() for n in
             json.load(open(f"perturbed_vocabs/{a.dataset}_plain.json"))]
    pools = [candidates(model, n) for n in plain]
    print("pool sizes:", [len(p) for p in pools], flush=True)

    def run(vocab, offset, limit):
        miou, _, _ = evaluate(model, samples, vocab, "none", 0.5, 336,
                              limit=limit, ignore=ignore, offset=offset,
                              log_every=10000)
        return miou * 100

    cur = list(plain)
    cur_score = run(cur, 0, 100)
    trace = [{"init": cur_score}]
    order = sorted(range(len(plain)), key=lambda i: plain[i])
    for i in order:
        if len(pools[i]) == 1:
            continue
        best_w, best_s = cur[i], cur_score
        for w in pools[i]:
            if w == cur[i]:
                continue
            trial = list(cur)
            trial[i] = w
            s = run(trial, 0, 100)
            if s < best_s:
                best_w, best_s = w, s
        cur[i], cur_score = best_w, best_s
        trace.append({"class": plain[i], "chosen": best_w,
                      "search_miou": cur_score})
        print(f"{plain[i]} -> {best_w} ({cur_score:.2f})", flush=True)

    held_ans = run(cur, 100, 200)
    held_plain = run(plain, 100, 200)
    syn100 = json.load(open(f"perturbed_vocabs/{a.dataset}_syn100_s0.json"))
    held_syn100 = run(syn100, 100, 200)
    res = {"prereg": "prereg_w4f_ans.md", "variant": a.variant,
           "dataset": a.dataset, "ans_vocab": cur,
           "search_miou_100": cur_score, "heldout_ans_200": held_ans,
           "heldout_plain_200": held_plain, "heldout_syn100_200": held_syn100,
           "trace": trace}
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("trace", "ans_vocab")}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
