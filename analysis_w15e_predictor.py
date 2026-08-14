"""W15-E (prereg_w15e_perclass_predictor.md): per-class harm predictor analysis."""
import json
import numpy as np
from scipy.stats import spearmanr
import torch
from clip_seg import DenseCLIP

R = "/media/dell/DATA/ovss/runs"
PV = "perturbed_vocabs"

plain = json.load(open(f"{PV}/voc21_plain.json"))
vabs = json.load(open(f"{PV}/voc21_plain_vabs64.json"))
# VABS negatives are comma-joined aliases of the background entry (row 0)
negs = [s.strip() for s in vabs[0].split(",")[1:]]
assert len(negs) == 64, len(negs)
print("negatives:", len(negs))

model = DenseCLIP("sclip", device="cuda" if torch.cuda.is_available() else "cpu")
E_cls = model.encode_text_raw([f"a photo of a {n}." for n in plain])
E_neg = model.encode_text_raw([f"a photo of a {n}." for n in negs])
maxcos = (E_cls @ E_neg.T).max(1).values.tolist()

out = {"prereg": "prereg_w15e_perclass_predictor.md", "negatives": len(negs),
       "variants": {}}
for v in ("sclip", "naclip"):
    reva = json.load(open(f"{R}/w15_fullrand_s0_{v}.json"))["arms"]["sam_reg_vabs"]["per_class"]
    base = json.load(open(f"{R}/w4a_voc21full_{v}_plain.json"))["per_class"]
    ks = [str(i) for i in range(1, 21)]  # foreground only
    delta = [(reva[k] - base[k]) * 100 for k in ks]
    pred = [maxcos[int(k)] for k in ks]
    rho, p = spearmanr(pred, delta)
    out["variants"][v] = {"rho": rho, "p": p,
                          "per_class": {plain[int(k)]: {"delta": round(d, 2),
                                                        "maxcos": round(c, 4)}
                                        for k, d, c in zip(ks, delta, pred)}}
    print(v, "rho=%.3f p=%.4f" % (rho, p))

json.dump(out, open(f"{R}/w15e_predictor.json", "w"), indent=1)
