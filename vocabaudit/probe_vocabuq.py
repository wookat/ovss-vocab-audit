"""W2c: VocabUQ go/no-go (stage12_month/prereg_w2c_vocabuq.md).

Split conformal over SAM regions. Score: s(r,c) = 1 - mean_v p_v(c|r)
+ lambda*std_v p_v(c|r) over vocabulary variants (plain + 3 syn100 seeds).
Calibrate on COCO-Object regions, test on VOC-21 / Context-60 regions.
"""
import argparse, json, math
import numpy as np
import torch
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, resize_short, to_tensor
from probe_d1sam import build_region_map

LAM = 1.0
ALPHA = 0.1
MIN_PX = 100
MAJ_FRAC = 0.5


def dense_probs(model, t, T, qi, h0, w0, K, scale=40.0):
    import torch.nn.functional as F
    from eval_seg import seg_logits
    logits = seg_logits(model, t, T, window=224, stride=112, logit_scale=scale)
    p = logits.softmax(0)  # (Q,H,W)
    Kq = int(qi.max()) + 1
    if Kq != p.shape[0]:
        agg = torch.zeros(Kq, *p.shape[1:], device=p.device)
        agg.index_add_(0, qi.to(p.device), p)
        p = agg
    p = F.interpolate(p.unsqueeze(0), size=(h0, w0), mode="bilinear",
                      align_corners=False)[0]
    return p  # Kq,h0,w0


def collect_regions(model, gen, samples, vocabs, K, ignore, quota, scale=40.0):
    """Return per-region: mean_p (K,), std_p (K,), true class, per-variant p (V,K)."""
    Ts = []
    for names in vocabs:
        T, qi = class_embeddings(model, names)
        Ts.append((T.to(model.device), qi))
    rows = []
    for ip, gp, loader in samples:
        if len(rows) >= quota:
            break
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        ps = [dense_probs(model, t, T, qi, h0, w0, K, scale)[:K] for T, qi in Ts]
        gt_r = np.array(Image.fromarray(gt.astype(np.int32), mode="I").resize(
            (w0, h0), Image.NEAREST))
        masks = gen.generate(np.array(img))
        reg = build_region_map(masks, h0, w0)
        for r in range(reg.max() + 1):
            sel_np = reg == r
            if sel_np.sum() < MIN_PX:
                continue
            g = gt_r[sel_np]
            g = g[(g != ignore) & (g >= 0) & (g < K)]
            if len(g) < MAJ_FRAC * sel_np.sum():
                continue
            vals, cnts = np.unique(g, return_counts=True)
            if cnts.max() < MAJ_FRAC * len(g):
                continue
            y = int(vals[cnts.argmax()])
            sel = torch.from_numpy(sel_np).to(model.device)
            rp = torch.stack([p[:, sel].mean(1) for p in ps])  # V,K
            rows.append((rp.cpu().numpy(), y))
            if len(rows) >= quota:
                break
    return rows


def scores(rp):
    mean = rp.mean(0)
    std = rp.std(0)
    return 1.0 - mean + LAM * std  # per class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-ckpt", default="/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)
    model = DenseCLIP("sclip", device="cuda")

    def vocabs_for(ds, plain):
        vs = [list(plain)]
        for s in range(3):
            vs.append(json.load(open(f"perturbed_vocabs/{ds}_syn50_s{s}.json")))
        return vs

    # calibration: COCO-Object 0-99
    samples, names, ignore = data.DATASETS["cocoobj"]()
    Kc = len(names)
    cal = collect_regions(model, gen, samples[:100], vocabs_for("cocoobj", names),
                          Kc, ignore, quota=500)
    cal_s = np.array([scores(rp)[y] for rp, y in cal])
    n = len(cal_s)
    q = float(np.sort(cal_s)[min(math.ceil((n + 1) * (1 - ALPHA)) - 1, n - 1)])
    # per-variant thresholds (single-vocab scores)
    qv = []
    for v in range(4):
        sv = np.array([1.0 - rp[v][y] for rp, y in cal])
        qv.append(float(np.sort(sv)[min(math.ceil((n + 1) * (1 - ALPHA)) - 1, n - 1)]))
    res = {"prereg": "stage12_month/prereg_w2c_vocabuq.md", "n_cal": n,
           "q_ensemble": q, "q_variants": qv, "lambda": LAM, "alpha": ALPHA}

    for ds, off, lim in [("voc21", 300, 100), ("ctx60", 0, 100)]:
        samples, names, ignore = data.DATASETS[ds]()
        K = len(names)
        rows = collect_regions(model, gen, samples[off:off + lim],
                               vocabs_for(ds, names), K, ignore, quota=300)
        cov = szs = 0
        cov_v = [0] * 4
        flag_harm = harm = 0
        for rp, y in rows:
            s = scores(rp)
            inset = s <= q
            cov += int(inset[y])
            szs += int(inset.sum())
            for v in range(4):
                cov_v[v] += int(1.0 - rp[v][y] <= qv[v])
            if ds == "voc21" and names[y].split(",")[0] in ("person", "tvmonitor"):
                if int(rp[0].argmax()) != y:  # plain-vocab region error
                    harm += 1
                    if inset.sum() >= 4:
                        flag_harm += 1
        m = len(rows)
        res[ds] = {"n_test": m, "coverage": cov / m, "mean_set_size": szs / m,
                   "set_frac_of_vocab": szs / m / K,
                   "coverage_per_variant": [c / m for c in cov_v]}
        if ds == "voc21":
            res[ds]["person_tv_err_regions"] = harm
            res[ds]["person_tv_err_flagged"] = flag_harm
        print(ds, json.dumps(res[ds]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
