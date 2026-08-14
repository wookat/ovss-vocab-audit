"""C6 round-2: label-free per-region config routing (stage11_phase3/prereg_c6r2_c4.md).

Pool = top-4 configs by label-free whole-output margin (E2). Gates G1/G2/G3.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from probe_headlens_e2 import HeadLensL
from probe_d1sam import build_region_map
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter

POOL = [("qq", 12), ("ident", 12), ("vanilla", 12), ("kk", 12)]  # frozen, label-free


@torch.no_grad()
def class_probs(emb, gh, gw, T, qi, h0, w0, K, scale=40.0):
    feat = F.normalize(emb.float(), dim=-1)[0]
    sims = (feat @ T.T).reshape(gh, gw, -1).permute(2, 0, 1)
    sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                         align_corners=False)[0]
    probs = (scale * sims).softmax(0)
    pooled = torch.zeros(K, h0, w0, device=probs.device)
    idx = qi.to(probs.device).view(-1, 1, 1).expand_as(probs)
    pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vocab", default="perturbed_vocabs/voc21_official.json")
    ap.add_argument("--sam-ckpt", default="/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    model = HeadLensL()
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    C = len(POOL)
    keys = [f"{f}_L{L}" for f, L in POOL]
    Hn = 12

    meters = {k: IoUMeter(K, ignore) for k in keys}
    gates = {g: IoUMeter(K, ignore) for g in ["G1", "G2", "G3"]}
    m_oracle = IoUMeter(K, ignore)

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        probs, preds, gmarg = [], [], []
        for (f, L), k in zip(POOL, keys):
            outs, bias, (gh, gw) = model.head_outputs_at(t, f, L)
            emb = model.subset_emb(outs, bias, range(Hn))
            p = class_probs(emb, gh, gw, T, qi, h0, w0, K)
            top2 = p.topk(2, dim=0).values
            gmarg.append(float((top2[0] - top2[1]).mean()))
            probs.append(p)
            pr = p.argmax(0).cpu().numpy()
            preds.append(pr)
            meters[k].update(pr, gt)
        preds = np.stack(preds)
        fallback = int(np.argmax(gmarg))

        gt_r = np.array(Image.fromarray(gt.astype(np.int32), mode="I").resize(
            (w0, h0), Image.NEAREST))
        valid = (gt_r != ignore) & (gt_r >= 0) & (gt_r < K)
        correct = (preds == gt_r[None]) & valid[None]

        masks = gen.generate(np.array(img))
        reg = build_region_map(masks, h0, w0)
        outs_g = {g: preds[fallback].copy() for g in gates}
        out_or = preds[fallback].copy()
        for r in range(reg.max() + 1):
            sel_np = reg == r
            if not sel_np.any():
                continue
            sel = torch.from_numpy(sel_np).to(model.device)
            marg, conf, labs = [], [], []
            for c in range(C):
                p = probs[c][:, sel]
                top2 = p.topk(2, dim=0).values
                marg.append(float((top2[0] - top2[1]).mean()))
                conf.append(float(top2[0].mean()))
                labs.append(int(p.mean(1).argmax()))
            c1, c2 = int(np.argmax(marg)), int(np.argmax(conf))
            vals, cnts = np.unique(labs, return_counts=True)
            maj = vals[cnts.argmax()]
            agree = [c for c in range(C) if labs[c] == maj]
            c3 = agree[int(np.argmax([marg[c] for c in agree]))] if agree else c1
            outs_g["G1"][sel_np] = preds[c1][sel_np]
            outs_g["G2"][sel_np] = preds[c2][sel_np]
            outs_g["G3"][sel_np] = preds[c3][sel_np]
            acc = correct[:, sel_np].sum(1)
            out_or[sel_np] = preds[int(acc.argmax())][sel_np]
        for g in gates:
            gates[g].update(outs_g[g], gt_r)
        m_oracle.update(out_or, gt_r)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}] " + " ".join(
                f"{g}={gates[g].miou()[0]*100:.2f}" for g in gates) +
                f" oracle={m_oracle.miou()[0]*100:.2f}", flush=True)

    res = {"prereg": "stage11_phase3/prereg_c6r2_c4.md",
           "pool": keys,
           "pool_miou": {k: meters[k].miou()[0] * 100 for k in keys},
           "gates": {g: gates[g].miou()[0] * 100 for g in gates},
           "pool_oracle": m_oracle.miou()[0] * 100,
           "best_single_full20": 53.2278}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
