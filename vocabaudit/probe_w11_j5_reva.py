"""W11-J5 composability cell: REVA (VABS64 + SAM arbitration) with and
without OWLv2 vocabulary pruning. SCLIP / VOC test-300."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

import data
from clip_seg import DenseCLIP
from eval_seg import IoUMeter, class_embeddings, resize_short, to_tensor, seg_logits
from probe_d1sam import build_region_map, pool_regions

THRESH = 0.2
SCALE = 40.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip")
    ap.add_argument("--vabs-vocab", required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    samples = samples[:a.limit]
    K = len(plain_names)
    labels = [n.split(",")[0].strip() for n in plain_names]

    vabs_names = json.load(open(a.vabs_vocab))
    T_vabs, qi_vabs = class_embeddings(model, vabs_names)
    T_vabs = T_vabs.to(model.device)

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to("cuda").eval()
    queries = [f"a photo of a {l}" for l in labels[1:]]

    meters = {k: IoUMeter(K, ignore) for k in
              ["reva", "reva_prune"]}
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T_vabs)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        k_out = int(qi_vabs.max()) + 1
        probs = (SCALE * sims).softmax(0)
        pooled = torch.zeros(k_out, h0, w0, device=sims.device)
        idx = qi_vabs.to(sims.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)

        img_np = np.asarray(img.resize((w0, h0)))
        masks = gen.generate(img_np)
        reg = build_region_map(masks, h0, w0)
        meters["reva"].update(pool_regions(pooled, reg), gt)

        inputs = proc(text=[queries], images=img, padding="max_length",
                      return_tensors="pt").to("cuda")
        det = proc.post_process_object_detection(
            owl(**inputs), threshold=THRESH,
            target_sizes=torch.tensor([(h0, w0)]))[0]
        supported = np.zeros(k_out, dtype=bool)
        supported[0] = True
        for li in det["labels"].tolist():
            if li + 1 < K:
                supported[li + 1] = True
        pp = pooled.clone()
        pp[torch.from_numpy(~supported).to(pp.device)] = -1e4
        meters["reva_prune"].update(pool_regions(pp, reg), gt)

        if (i + 1) % 25 == 0:
            print(f"[{i+1}] reva={meters['reva'].miou()[0]*100:.2f} "
                  f"prune={meters['reva_prune'].miou()[0]*100:.2f}", flush=True)

    res = {"prereg": "prereg_w11_j5_vocabprune.md", "variant": a.variant}
    for k, m in meters.items():
        res[k] = m.miou()[0] * 100
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
