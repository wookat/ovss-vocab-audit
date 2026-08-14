"""prereg_v4: D1-SAM round -- SAM automatic masks replace SLIC regions.

Arms: pix_vabs, sam_reg_vabs, sam_reg_rand. Uncovered pixels fall back to the
pixel-level prediction of the same arm's vocabulary.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter




def build_region_map(masks, H, W):
    """Larger masks first; smaller masks overwrite (preserve detail). -1 = uncovered."""
    reg = np.full((H, W), -1, dtype=np.int64)
    for i, m in enumerate(sorted(masks, key=lambda x: -x["area"])):
        reg[m["segmentation"]] = i
    return reg


def pool_regions(probs, reg):
    """probs (K,H,W) gpu; reg (H,W) with -1 uncovered. Returns pred (H,W) np."""
    K = probs.shape[0]
    pix_pred = probs.argmax(0).cpu().numpy()
    if reg.max() < 0:
        return pix_pred
    flat = probs.reshape(K, -1)
    r = torch.from_numpy(reg.reshape(-1)).to(probs.device)
    valid = r >= 0
    R = int(r.max()) + 1
    out = torch.zeros(R, K, device=probs.device)
    cnt = torch.zeros(R, device=probs.device)
    out.index_add_(0, r[valid], flat[:, valid].T)
    cnt.index_add_(0, r[valid], torch.ones(int(valid.sum()), device=probs.device))
    reg_pred = (out / cnt.clamp_min(1).unsqueeze(1)).argmax(1).cpu().numpy()
    pred = pix_pred.copy()
    cov = reg >= 0
    pred[cov] = reg_pred[reg[cov]]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vabs-vocab", required=True)
    ap.add_argument("--rand-vocab", required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--skip-dev", action="store_true",
                    help="exclude images [300,400) of the split (dev-100)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points_per_side)

    if a.variant == "proxyclip":
        from proxyclip_seg import ProxyCLIP
        model = ProxyCLIP()
    elif a.variant == "lposs":
        from newgen_seg import LPOSS
        model = LPOSS()
    elif a.variant == "scclip":
        from newgen_seg import SCCLIP
        model = SCCLIP()
    else:
        model = DenseCLIP(a.variant, model_name=a.model)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    if a.skip_dev:
        samples = samples[:300] + samples[400:]
    K = len(plain_names)

    vabs_names = json.load(open(a.vabs_vocab))
    rand_names = json.load(open(a.rand_vocab))
    T_vabs, qi_vabs = class_embeddings(model, vabs_names)
    T_rand, qi_rand = class_embeddings(model, rand_names)
    T_vabs, T_rand = T_vabs.to(model.device), T_rand.to(model.device)

    # vocabs may append an extra background class (e.g. ADE-150 has none);
    # size meters to the prediction space, GT classes stay 0..K-1
    Km = max(K, int(qi_vabs.max()) + 1, int(qi_rand.max()) + 1)
    meters = {k: IoUMeter(Km, ignore) for k in
              ["pix_vabs", "sam_reg_vabs", "sam_reg_rand"]}
    scale = 40.0

    t_clip = t_sam = 0.0
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        t0 = time.time()
        T_all = torch.cat([T_vabs, T_rand], 0)
        sims = seg_logits(model, t, T_all)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        s_vabs, s_rand = sims[:T_vabs.shape[0]], sims[T_vabs.shape[0]:]

        def to_class(s, qi):
            k_out = int(qi.max()) + 1
            probs = (scale * s).softmax(0)
            pooled = torch.zeros(k_out, *s.shape[1:], device=s.device)
            idx = qi.to(s.device).view(-1, 1, 1).expand_as(probs)
            pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
            return pooled

        p_vabs, p_rand = to_class(s_vabs, qi_vabs), to_class(s_rand, qi_rand)
        meters["pix_vabs"].update(p_vabs.argmax(0).cpu().numpy(), gt)
        torch.cuda.synchronize(); t1 = time.time(); t_clip += t1 - t0

        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        meters["sam_reg_vabs"].update(pool_regions(p_vabs, reg), gt)
        meters["sam_reg_rand"].update(pool_regions(p_rand, reg), gt)
        torch.cuda.synchronize(); t_sam += time.time() - t1

        if (i + 1) % 10 == 0:
            print(f"[{i+1}] pix={meters['pix_vabs'].miou()[0]*100:.2f} "
                  f"sam={meters['sam_reg_vabs'].miou()[0]*100:.2f} "
                  f"samrand={meters['sam_reg_rand'].miou()[0]*100:.2f}", flush=True)

    res = {"prereg": "prereg_v5.md", "variant": a.variant, "dataset": a.dataset,
           "dev": {"offset": a.offset, "n": a.limit},
           "sam": {"ckpt": a.sam_ckpt, "points_per_side": a.points_per_side},
           "arms": {}, "skip_dev": bool(a.skip_dev),
           "timing_s_per_img": {"clip_pixel": t_clip / max(len(samples), 1),
                                "sam_and_pooling": t_sam / max(len(samples), 1)}}
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "miou_all": m.miou_all(), "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
