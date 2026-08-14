"""W1b: few-label per-region config router (stage12_month/prereg_w1_router.md).

Phase 1 (--phase train): on dev 1-50, evaluate all 20 configs, pick top-4 by mIoU,
collect (region, config) features + region pixel accuracy, fit ridge router.
Phase 2 (--phase eval): route on held-out dev 51-100 and test-300.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from probe_headlens import FLAVORS
from probe_headlens_e2 import HeadLensL
from probe_d1sam import build_region_map
from probe_route_lf import class_probs
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter


def region_features(probs, preds, sel, c, maj):
    p = probs[c][:, sel]
    top2 = p.topk(2, dim=0).values
    margin = float((top2[0] - top2[1]).mean())
    conf = float(top2[0].mean())
    ent = float(-(p * (p + 1e-9).log()).sum(0).mean())
    return [margin, conf, ent, float(np.log(int(sel.sum().item()))),
            1.0 if int(p.mean(1).argmax()) == maj else 0.0]


def run_images(model, gen, samples, pool, T, qi, K, ignore, w=None, collect=False):
    """If w given: route and return meters. If collect: return (X, y) per region."""
    Hn = 12
    C = len(pool)
    Xs, ys = [], []
    m_route = IoUMeter(K, ignore)
    meters = [IoUMeter(K, ignore) for _ in pool]
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        probs, preds, gmarg = [], [], []
        for f, L in pool:
            outs, bias, (gh, gw) = model.head_outputs_at(t, f, L)
            emb = model.subset_emb(outs, bias, range(Hn))
            p = class_probs(emb, gh, gw, T, qi, h0, w0, K)
            top2 = p.topk(2, dim=0).values
            gmarg.append(float((top2[0] - top2[1]).mean()))
            probs.append(p)
            preds.append(p.argmax(0).cpu().numpy())
        for c in range(C):
            meters[c].update(preds[c], gt)
        preds_np = np.stack(preds)
        gt_r = np.array(Image.fromarray(gt.astype(np.int32), mode="I").resize(
            (w0, h0), Image.NEAREST))
        valid = (gt_r != ignore) & (gt_r >= 0) & (gt_r < K)
        masks = gen.generate(np.array(img))
        reg = build_region_map(masks, h0, w0)
        out = preds_np[int(np.argmax(gmarg))].copy()
        for r in range(reg.max() + 1):
            sel_np = reg == r
            if not sel_np.any():
                continue
            sel = torch.from_numpy(sel_np).to(model.device)
            labs = [int(probs[c][:, sel].mean(1).argmax()) for c in range(C)]
            vals, cnts = np.unique(labs, return_counts=True)
            maj = int(vals[cnts.argmax()])
            feats = [region_features(probs, preds_np, sel, c, maj)
                     for c in range(C)]
            if collect:
                v = valid & sel_np
                if v.sum() == 0:
                    continue
                for c in range(C):
                    acc = float(((preds_np[c] == gt_r) & v).sum() / v.sum())
                    onehot = [0.0] * C; onehot[c] = 1.0
                    Xs.append(feats[c] + onehot)
                    ys.append(acc)
            if w is not None:
                scores = []
                for c in range(C):
                    onehot = [0.0] * C; onehot[c] = 1.0
                    x = np.array(feats[c] + onehot + [1.0])
                    scores.append(float(x @ w))
                out[sel_np] = preds_np[int(np.argmax(scores))][sel_np]
        if w is not None:
            m_route.update(out, gt_r)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}]", flush=True)
    if collect:
        return np.array(Xs), np.array(ys), meters
    return m_route, meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-ckpt", default="/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--vocab", default="perturbed_vocabs/voc21_official.json")
    ap.add_argument("--e2", default="/media/dell/DATA/ovss/runs/headlens_e2_dev100.json")
    ap.add_argument("--test-limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    model = HeadLensL()
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)

    train_s = samples[300:350]
    held_s = samples[350:400]
    test_s = samples[:a.test_limit]

    # pool selection: top-4 by mIoU on TRAIN images over all 20 configs
    cfgs = [(f, L) for f in FLAVORS for L in [8, 9, 10, 11, 12]]
    Hn = 12
    tr_meters = {c: IoUMeter(K, ignore) for c in range(len(cfgs))}
    for ip, gp, loader in train_s:
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        for ci, (f, L) in enumerate(cfgs):
            outs, bias, (gh, gw) = model.head_outputs_at(t, f, L)
            emb = model.subset_emb(outs, bias, range(Hn))
            p = class_probs(emb, gh, gw, T, qi, h0, w0, K)
            tr_meters[ci].update(p.argmax(0).cpu().numpy(), gt)
    tr_miou = {ci: tr_meters[ci].miou()[0] * 100 for ci in range(len(cfgs))}
    top4 = sorted(tr_miou, key=lambda c: -tr_miou[c])[:4]
    pool = [cfgs[c] for c in top4]
    print("pool:", pool, [round(tr_miou[c], 2) for c in top4], flush=True)

    # collect training regions + fit ridge
    X, y, _ = run_images(model, gen, train_s, pool, T, qi, K, ignore, collect=True)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    lam = 1.0
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    print(f"train regions: {len(y)}, ridge w: {np.round(w, 3).tolist()}", flush=True)

    res = {"prereg": "stage12_month/prereg_w1_router.md",
           "pool": [f"{f}_L{L}" for f, L in pool],
           "train_miou_20cfg": {f"{cfgs[c][0]}_L{cfgs[c][1]}": tr_miou[c]
                                for c in sorted(tr_miou, key=lambda c: -tr_miou[c])},
           "ridge_w": w.tolist()}

    for split, ss in [("held_dev", held_s), ("test300", test_s)]:
        m_route, meters = run_images(model, gen, ss, pool, T, qi, K, ignore, w=w)
        res[split] = {"routed": m_route.miou()[0] * 100,
                      "pool_members": {f"{f}_L{L}": m.miou()[0] * 100
                                       for (f, L), m in zip(pool, meters)}}
        print(split, json.dumps(res[split]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
