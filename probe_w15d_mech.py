"""W15-D (prereg_w15d_negclaim_mech.md): negative-claim accounting.

Same arms as probe_d1sam (region arbitration, vabs vs rand negatives); logs
background-claim precision/recall vs GT background for both arms, plus the
share of pixels whose winning row was a negative (rather than the plain
'background' row).
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor
from probe_d1sam import build_region_map, pool_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vabs-vocab", required=True)
    ap.add_argument("--rand-vocab", required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points_per_side)
    model = DenseCLIP(a.variant, model_name=a.model)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]

    stats = {arm: {"tp": 0, "fp": 0, "fn": 0, "neg_row_pixels": 0, "total": 0}
             for arm in ("vabs", "rand")}
    scale = 40.0

    vocabs = {"vabs": json.load(open(a.vabs_vocab)),
              "rand": json.load(open(a.rand_vocab))}
    embs = {k: class_embeddings(model, v) for k, v in vocabs.items()}

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        gtm = gt != ignore
        gt_bg = (gt == 0) & gtm

        for arm in ("vabs", "rand"):
            T, qi = embs[arm]
            T = T.to(model.device)
            sims = seg_logits(model, t, T)
            sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0),
                                 mode="bilinear", align_corners=False)[0]
            probs = (scale * sims).softmax(0)
            k_out = int(qi.max()) + 1
            pooled = torch.zeros(k_out, *sims.shape[1:], device=sims.device)
            idx = qi.to(sims.device).view(-1, 1, 1).expand_as(probs)
            pooled.scatter_reduce_(0, idx, probs, reduce="amax",
                                   include_self=False)
            pred = pool_regions(pooled, reg)
            # winning-row accounting at pixel level (pre-arbitration):
            row_win = sims.argmax(0).cpu().numpy()
            qi_np = qi.cpu().numpy()
            is_neg_row = np.zeros(len(qi_np), bool)
            for r_i, nm in enumerate(vocabs[arm]):
                if qi_np[r_i] == 0 and str(nm).split(",")[0].strip().lower() != "background":
                    is_neg_row[r_i] = True
            pred_bg = (pred == 0) & gtm
            s = stats[arm]
            s["tp"] += int((pred_bg & gt_bg).sum())
            s["fp"] += int((pred_bg & ~gt_bg & gtm).sum())
            s["fn"] += int((~pred_bg & gt_bg & gtm).sum())
            s["neg_row_pixels"] += int(is_neg_row[row_win][gtm].sum())
            s["total"] += int(gtm.sum())
        if (i + 1) % 25 == 0:
            print(f"[{i+1}]", {arm: round(v['tp'] / max(v['tp'] + v['fn'], 1), 4)
                               for arm, v in stats.items()}, flush=True)

    res = {"prereg": "prereg_w15d_negclaim_mech.md", "variant": a.variant,
           "dataset": a.dataset, "offset": a.offset, "limit": a.limit,
           "vabs_vocab": a.vabs_vocab, "rand_vocab": a.rand_vocab, "arms": {}}
    for arm, s in stats.items():
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        rec = s["tp"] / max(s["tp"] + s["fn"], 1)
        res["arms"][arm] = dict(s, bg_precision=prec, bg_recall=rec,
                                neg_row_share=s["neg_row_pixels"] / max(s["total"], 1))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {"prec": round(v["bg_precision"], 4),
                          "rec": round(v["bg_recall"], 4),
                          "negshare": round(v["neg_row_share"], 4)}
                      for k, v in res["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
