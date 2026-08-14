"""W4c E1: graph-spectral decomposition of vocabulary-perturbation noise.

On a ClearCLIP base, compute per-window class logits under plain vs syn100
vocab, project the logit difference onto the eigenbasis of the DINO kNN graph
Laplacian, and measure (i) the energy fraction in the high-frequency half and
(ii) the attenuation ratio of propagation on the high vs low band.
PASS per prereg_w4c_propsmooth.md: >=60% high-band energy and >=2x relative
attenuation.
"""
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, resize_short, to_tensor
from newgen_seg import dino_affinity, propagate
import data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, names, ignore = data.DATASETS[a.dataset]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DenseCLIP("clearclip", device=dev)
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                          skip_validation=True).to(dev).eval()
    vocabs = {}
    for vk in ("plain", "syn100_s0"):
        vnames = json.load(open(f"perturbed_vocabs/{a.dataset}_{vk}.json"))
        emb, qi = class_embeddings(model, vnames, "none")
        vocabs[vk] = (emb.to(dev), qi.to(dev), len(vnames))

    hi_frac, att_ratio = [], []
    with torch.no_grad():
        for img_path, _, _ in samples[:a.n]:
            img = Image.open(img_path).convert("RGB")
            img_r, _ = resize_short(img, 224)
            t = to_tensor(img_r, dev)[:, :, :224, :224]
            feat, (gh, gw) = model.encode_dense(t)
            feat = F.normalize(feat.float(), dim=-1)
            logits = {}
            for vk, (emb, qi, K) in vocabs.items():
                ql = feat @ emb.T.float()  # B,N,Q
                cl = torch.zeros(*ql.shape[:-1], K, device=dev)
                cnt = torch.zeros(K, device=dev)
                cl.index_add_(-1, qi, ql)
                cnt.index_add_(0, qi, torch.ones(qi.shape[0], device=dev))
                logits[vk] = cl / cnt.clamp_min(1)
            diff = (logits["plain"] - logits["syn100_s0"])[0]
            Sk = dino_affinity(dino, t, 32)[0]
            L = torch.eye(Sk.shape[0], device=dev) - 0.5 * (Sk + Sk.T)
            evals, evecs = torch.linalg.eigh(L)
            coef = evecs.T @ diff  # N x K spectral coefficients
            N = coef.shape[0]
            lo, hi = coef[: N // 2], coef[N // 2:]
            e_lo, e_hi = (lo ** 2).sum().item(), (hi ** 2).sum().item()
            hi_frac.append(e_hi / max(e_lo + e_hi, 1e-8))
            pdiff = propagate(Sk.unsqueeze(0), diff.unsqueeze(0), 0.9, 10)[0]
            pcoef = evecs.T @ pdiff
            plo, phi = pcoef[: N // 2], pcoef[N // 2:]
            att_hi = e_hi / max((phi ** 2).sum().item(), 1e-8)
            att_lo = e_lo / max((plo ** 2).sum().item(), 1e-8)
            att_ratio.append(att_hi / max(att_lo, 1e-8))
    res = {"n": a.n,
           "high_energy_frac_mean": sum(hi_frac) / len(hi_frac),
           "attenuation_ratio_mean": sum(att_ratio) / len(att_ratio),
           "per_image_high_frac": hi_frac, "per_image_att_ratio": att_ratio}
    print(json.dumps({k: v for k, v in res.items() if "per_image" not in k}))
    json.dump(res, open(a.out, "w"), indent=1)
