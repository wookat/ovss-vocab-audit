"""LexRO stage A: offline teacher cache on ADE train images (prereg_lexro_v1.md).

Per image (224 centre crop of 336 short-side):
- NACLIP dense patch features (196, 512) f16
- REVA teacher labels per patch: pixel probs (COCO-171 plain + VABS-64 negs folded
  into a 172nd bg class) -> SAM region pooling -> patch-grid argmax label (uint8)
  + confidence (f16).
Saved as sharded .pt files.
"""
import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from clip_seg import DenseCLIP
from eval_seg import class_embeddings, resize_short, to_tensor
from probe_d1sam import build_region_map, pool_regions

ADE_TRAIN = "/media/dell/DATA/ovss/datasets/ADEChallengeData2016/images/training"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", default="perturbed_vocabs/coco171_plain_vabs64.json")
    ap.add_argument("--sam-ckpt", default="/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    files = sorted(os.listdir(ADE_TRAIN))
    rng = random.Random(a.seed)
    files = rng.sample(files, min(a.n, len(files)))
    json.dump(files, open(f"{a.out_dir}/images.json", "w"))

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    model = DenseCLIP("naclip")
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    K = int(qi.max()) + 1  # 171 classes + bg(=last, holds VABS negatives)
    scale = 40.0

    shard_feats, shard_lbl, shard_conf, shard_ids = [], [], [], []
    si = a.start // a.shard_size
    for i, fn in enumerate(files):
        if i < a.start:
            continue
        img = Image.open(f"{ADE_TRAIN}/{fn}").convert("RGB")
        img_r, _ = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        _, _, H, W = t.shape
        y, x = (H - 224) // 2, (W - 224) // 2
        crop = t[:, :, y:y + 224, x:x + 224]
        feat, (gh, gw) = model.encode_dense(crop)
        feat = F.normalize(feat[0].float(), dim=-1)  # (196,512)
        sims = feat @ T.T
        probs = (scale * sims).softmax(-1)  # (P,Q)
        pooled = torch.zeros(feat.shape[0], K, device=feat.device)
        pooled.scatter_reduce_(1, qi.to(feat.device).unsqueeze(0).expand_as(probs),
                               probs, reduce="amax", include_self=False)
        # upsample to pixel grid for SAM pooling
        pk = pooled.T.reshape(K, gh, gw)
        pk = F.interpolate(pk.unsqueeze(0), size=(224, 224), mode="bilinear",
                           align_corners=False)[0]
        img_np = np.asarray(img_r)[y:y + 224, x:x + 224]
        masks = gen.generate(img_np)
        reg = build_region_map(masks, 224, 224)
        pred = pool_regions(pk, reg)  # (224,224) np
        # confidence: mean max-prob within each patch cell after pooling
        pred_t = torch.from_numpy(pred).to(feat.device)
        conf_pix = pk.max(0).values  # (224,224)
        cell = 224 // gh
        lbl_patch = pred_t.reshape(gh, cell, gw, cell).permute(0, 2, 1, 3).reshape(gh * gw, -1)
        lbl_mode = lbl_patch.mode(dim=1).values  # (P,)
        conf_patch = conf_pix.reshape(gh, cell, gw, cell).permute(0, 2, 1, 3).reshape(gh * gw, -1).mean(1)

        shard_feats.append(feat.half().cpu())
        shard_lbl.append(lbl_mode.to(torch.uint8).cpu())
        shard_conf.append(conf_patch.half().cpu())
        shard_ids.append(fn)
        if len(shard_ids) == a.shard_size or i == len(files) - 1:
            torch.save({"feat": torch.stack(shard_feats), "lbl": torch.stack(shard_lbl),
                        "conf": torch.stack(shard_conf), "ids": shard_ids},
                       f"{a.out_dir}/shard_{si:03d}.pt")
            print(f"shard {si} written ({i+1}/{len(files)})", flush=True)
            shard_feats, shard_lbl, shard_conf, shard_ids = [], [], [], []
            si += 1


if __name__ == "__main__":
    main()
