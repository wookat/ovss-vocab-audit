"""Same-protocol control: Trident's prompt-based SAM refinement stage applied
to our dense per-pixel class probabilities (same SAM ViT-B checkpoint as REVA).

Arms: pix (pixel baseline), trident_sam (Trident refinement).
Trident defaults: coarse_thresh=0.10, minimal_area=225, sam_mask_coff=0.005,
sam_iou_thresh=0.9 (their released config uses SAM ViT-H; we use ViT-B to match
REVA's compute -- disclosed in the paper).
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from trident_refine import sam_refinement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="naclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--skip-dev", action="store_true")
    ap.add_argument("--coarse-thresh", type=float, default=0.10)
    ap.add_argument("--minimal-area", type=int, default=225)
    ap.add_argument("--sam-mask-coff", type=float, default=0.005)
    ap.add_argument("--sam-iou-thresh", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    predictor = SamPredictor(sam)

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    if a.skip_dev:
        samples = samples[:300] + samples[400:]
    K = len(plain_names)

    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    meters = {k: IoUMeter(K, ignore) for k in ["pix", "trident_sam"]}
    scale = 40.0
    t_ref = 0.0

    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        probs = (scale * sims).softmax(0)
        pooled = torch.zeros(K, *sims.shape[1:], device=sims.device)
        idx = qi.to(sims.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
        pix_pred = pooled.argmax(0, keepdim=True)
        meters["pix"].update(pix_pred[0].cpu().numpy(), gt)

        t0 = time.time()
        img_np = np.asarray(img.resize((w0, h0)))
        predictor.set_image(img_np)
        refined, _, _, _ = sam_refinement(
            img_np, pix_pred, pooled, K, predictor,
            coarse_thresh=a.coarse_thresh, minimal_area=a.minimal_area,
            sam_mask_coff=a.sam_mask_coff, sam_iou_thresh=a.sam_iou_thresh)
        torch.cuda.synchronize(); t_ref += time.time() - t0
        meters["trident_sam"].update(refined[0].cpu().numpy(), gt)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}] pix={meters['pix'].miou()[0]*100:.2f} "
                  f"trident={meters['trident_sam'].miou()[0]*100:.2f}", flush=True)

    res = {"variant": a.variant, "dataset": a.dataset, "vocab": a.vocab,
           "skip_dev": bool(a.skip_dev),
           "trident": {"coarse_thresh": a.coarse_thresh, "minimal_area": a.minimal_area,
                       "sam_mask_coff": a.sam_mask_coff, "sam_iou_thresh": a.sam_iou_thresh,
                       "sam": "vit_b"},
           "timing_s_per_img": {"sam_refinement": t_ref / max(len(samples), 1)}, "arms": {}}
    for k, m in meters.items():
        miou, per = m.miou()
        res["arms"][k] = {"miou": miou, "miou_all": m.miou_all(), "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["miou"] * 100, 2) for k, v in res["arms"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
