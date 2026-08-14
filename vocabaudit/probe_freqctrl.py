"""W7a (prereg_w7a_freqctrl.md, frozen): build token-matched random-synonym
control vocabularies for the ANS cross-paradigm transfer claim.

For each class where the ANS vocabulary changed the name, sample a different
WordNet synonym from the same candidate pool (cosine [0.70,0.95]), matched on
CLIP BPE token count (+/-1; else nearest). 3 seeds. Writes control vocab
JSONs and a token-count comparison table.
"""
import argparse
import json
import random
import open_clip

from clip_seg import DenseCLIP
from probe_ans import candidates


def ntok(tokenizer, w):
    return int((tokenizer([w]) != 0).sum()) - 2  # minus SOT/EOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ans-run", default="/media/dell/DATA/ovss/runs/w4f_ans_clearclip.json")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--out-prefix", default="perturbed_vocabs/voc21_freqctrl")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    model = DenseCLIP("clearclip")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    plain = [n.split(",")[0].strip() for n in
             json.load(open(f"perturbed_vocabs/{a.dataset}_plain.json"))]
    ans = json.load(open(a.ans_run))["ans_vocab"]
    pools = [candidates(model, n) for n in plain]

    report = {"prereg": "prereg_w7a_freqctrl.md", "classes": [], "seeds": {}}
    for seed in range(3):
        rng = random.Random(seed)
        ctrl = []
        for c, (p, chosen, pool) in enumerate(zip(plain, ans, pools)):
            if chosen == p:
                ctrl.append(p)
                continue
            t_ans = ntok(tokenizer, chosen)
            alts = [w for w in pool[1:] if w != chosen]
            if not alts:
                ctrl.append(chosen)  # disclosed: no alternative
                if seed == 0:
                    report["classes"].append({"class": p, "ans": chosen,
                                              "ctrl": "NO_ALT"})
                continue
            matched = [w for w in alts if abs(ntok(tokenizer, w) - t_ans) <= 1]
            cand = matched if matched else \
                sorted(alts, key=lambda w: abs(ntok(tokenizer, w) - t_ans))[:1]
            pick = rng.choice(cand)
            ctrl.append(pick)
            if seed == 0:
                report["classes"].append({
                    "class": p, "ans": chosen, "ans_tok": t_ans,
                    "ctrl_example": pick, "ctrl_tok": ntok(tokenizer, pick),
                    "pool": alts})
        fp = f"{a.out_prefix}_s{seed}.json"
        json.dump(ctrl, open(fp, "w"))
        report["seeds"][f"s{seed}"] = ctrl
        print(fp, "written")
    json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
