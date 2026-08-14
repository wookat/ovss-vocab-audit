"""Pre-registered day-1 falsification probes for method line phase 2 (prereg_v3.md).

D1-0: oracle-region margin AUC for stolen classes (person, tvmonitor).
D1-1: SLIC region pooling vs pixel argmax under the same VABS64 vocabulary.
C3  : NullAbstain -- region-level p-values against a matched real-noun null
      lexicon with per-image BH-FDR, abstain -> background; control arm is the
      same region pooling with random-64 negative sink.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.segmentation import slic

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter

DEV_OFFSET, DEV_N = 300, 100
PERSON, TV = 15, 20


def region_labels(img_np, n_segments=200):
    return slic(img_np, n_segments=n_segments, compactness=10, start_label=0)


def pool_by_region(arr, regions):
    """arr: (K,H,W) tensor; regions: (H,W) int array. Returns (R,K)."""
    K = arr.shape[0]
    flat = arr.reshape(K, -1)
    r = torch.from_numpy(regions.reshape(-1)).to(arr.device)
    R = int(r.max()) + 1
    out = torch.zeros(R, K, device=arr.device)
    cnt = torch.zeros(R, device=arr.device)
    out.index_add_(0, r, flat.T)
    cnt.index_add_(0, r, torch.ones_like(r, dtype=torch.float))
    return out / cnt.clamp_min(1).unsqueeze(1)


@torch.no_grad()
def image_sims(model, img_path, T_all, short=336):
    img = Image.open(img_path).convert("RGB")
    img_r, (w0, h0) = resize_short(img, short)
    t = to_tensor(img_r, model.device)
    sims = seg_logits(model, t, T_all)  # (Q,H,W) cosine sims
    sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                         align_corners=False)[0]
    return sims, np.asarray(img.resize((w0, h0)))


def bh_fdr(p, q=0.05):
    """p: 1D np array. Returns boolean significance mask (BH)."""
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thresh
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--vabs-vocab", required=True, help="plain+VABS64 vocab json")
    ap.add_argument("--rand-vocab", required=True, help="plain+random64 vocab json")
    ap.add_argument("--null-vocab", required=True, help="null noun list json")
    ap.add_argument("--fdr-q", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.voc21()
    samples = samples[DEV_OFFSET:DEV_OFFSET + DEV_N]
    K = len(plain_names)

    vabs_names = json.load(open(a.vabs_vocab))
    rand_names = json.load(open(a.rand_vocab))
    null_words = json.load(open(a.null_vocab))
    # VABS/random vocabs fold negatives as comma sub-queries of class 0 (background)
    assert len(vabs_names) == K and len(rand_names) == K

    T_vabs, qi_vabs = class_embeddings(model, vabs_names)
    T_rand, qi_rand = class_embeddings(model, rand_names)
    T_tgt, _ = class_embeddings(model, plain_names)
    T_null, _ = class_embeddings(model, null_words)
    T_vabs, T_rand = T_vabs.to(model.device), T_rand.to(model.device)
    T_tgt, T_null = T_tgt.to(model.device), T_null.to(model.device)
    nq_v = T_vabs.shape[0]

    # D1-0 accumulators: margins for GT target regions vs GT background regions
    marg = {PERSON: {"pos": [], "bg": []}, TV: {"pos": [], "bg": []}}

    meters = {k: IoUMeter(K, ignore) for k in
              ["pix_vabs", "reg_vabs", "reg_rand", "c3"]}

    scale = 40.0
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        # one dense pass with the union of all query sets (concat)
        T_all = torch.cat([T_vabs, T_rand, T_null], 0)
        sims, img_np = image_sims(model, ip, T_all)
        s_vabs = sims[:nq_v]
        s_rand = sims[nq_v:nq_v + T_rand.shape[0]]
        s_null = sims[nq_v + T_rand.shape[0]:]

        def to_class(s, qi, k_out):
            probs = (scale * s).softmax(0)
            pooled = torch.zeros(k_out, *s.shape[1:], device=s.device)
            idx = qi.to(s.device).view(-1, 1, 1).expand_as(probs)
            pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
            return pooled

        kv = int(qi_vabs.max()) + 1
        kr = int(qi_rand.max()) + 1
        p_vabs = to_class(s_vabs, qi_vabs, kv)
        p_rand = to_class(s_rand, qi_rand, kr)

        # arm pix_vabs
        pred = p_vabs.argmax(0).cpu().numpy()
        pred[pred >= K] = 0
        meters["pix_vabs"].update(pred, gt)

        # regions
        regions = region_labels(img_np)
        Rv = pool_by_region(p_vabs, regions)  # (R, kv)
        Rr = pool_by_region(p_rand, regions)
        reg_pred_v = Rv.argmax(1).cpu().numpy()
        reg_pred_v[reg_pred_v >= K] = 0
        pred = reg_pred_v[regions]
        meters["reg_vabs"].update(pred, gt)
        reg_pred_r = Rr.argmax(1).cpu().numpy()
        reg_pred_r[reg_pred_r >= K] = 0
        meters["reg_rand"].update(reg_pred_r[regions], gt)

        # D1-0 oracle margins: GT connected components, region-avg raw sims,
        # margin = max target-query sim - max negative-query sim
        qi_np = qi_vabs.numpy()
        tgt_cols = {c: np.where(qi_np == c)[0] for c in (PERSON, TV)}
        neg_cols = np.where(qi_np == 0)[0][1:]  # background sub-queries = negatives
        sims_np = s_vabs  # (nq_v, H, W) on gpu

        def comp_margin(mask, cols):
            m = torch.from_numpy(mask).to(sims_np.device)
            v = sims_np[:, m].mean(1)  # (nq_v,)
            return float(v[cols].max() - v[neg_cols].max())

        for c in (PERSON, TV):
            lab, n = ndimage.label(gt == c)
            for j in range(1, n + 1):
                mask = lab == j
                if mask.sum() >= 100:
                    marg[c]["pos"].append(comp_margin(mask, tgt_cols[c]))
        lab_bg, n_bg = ndimage.label(gt == 0)
        for j in range(1, n_bg + 1):
            mask = lab_bg == j
            if mask.sum() >= 100:
                for c in (PERSON, TV):
                    marg[c]["bg"].append(comp_margin(mask, tgt_cols[c]))

        # region sims for C3
        s_t = pool_by_region(s_vabs, regions)  # (R, nq_v)

        # C3: region p-values vs null
        # target sims: reuse s_vabs first-K target queries (plain names, 1 query/class)
        tq = [np.where(qi_np == c)[0][0] for c in range(K)]
        s_tgt_r = s_t[:, tq].cpu().numpy()          # (R, K)
        s_null_r = pool_by_region(s_null, regions).cpu().numpy()  # (R, Nnull)
        pvals = (s_null_r[:, None, :] >= s_tgt_r[:, :, None]).mean(2)  # (R, K)
        sig = bh_fdr(pvals.reshape(-1), a.fdr_q).reshape(pvals.shape)
        reg_pred = np.zeros(pvals.shape[0], dtype=np.int64)
        for r in range(pvals.shape[0]):
            cand = np.where(sig[r, 1:])[0] + 1  # never "significant background"
            if len(cand):
                reg_pred[r] = cand[np.argmax(s_tgt_r[r, cand])]
        meters["c3"].update(reg_pred[regions], gt)

        if (i + 1) % 20 == 0:
            print(f"[{i+1}] pix={meters['pix_vabs'].miou()[0]*100:.2f} "
                  f"reg={meters['reg_vabs'].miou()[0]*100:.2f} "
                  f"c3={meters['c3'].miou()[0]*100:.2f}", flush=True)

    def auc(pos, neg):
        if not pos or not neg:
            return None
        x = np.array(pos + neg)
        y = np.array([1] * len(pos) + [0] * len(neg))
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        rp = ranks[y == 1].sum()
        n1, n0 = len(pos), len(neg)
        return float((rp - n1 * (n1 + 1) / 2) / (n1 * n0))

    res = {
        "prereg": "prereg_v3.md",
        "dev": {"offset": DEV_OFFSET, "n": DEV_N},
        "d1_0_auc": {"person": auc(marg[PERSON]["pos"], marg[PERSON]["bg"]),
                     "tvmonitor": auc(marg[TV]["pos"], marg[TV]["bg"]),
                     "n_regions": {str(c): {k: len(v) for k, v in marg[c].items()}
                                   for c in marg}},
        "arms": {},
    }
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                     indent=1))
    print("AUC:", res["d1_0_auc"])


if __name__ == "__main__":
    main()
