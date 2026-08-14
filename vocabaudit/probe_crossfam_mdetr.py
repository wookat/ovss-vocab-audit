"""W7b lineage go/no-go (prereg_w7b_lineage.md): MDETR boxes + SAM masks.

Third model for the BERT/RoBERTa-grounded lineage (GLIP unavailable in the
offline transformers stack; substitution to MDETR recorded in the prereg
before results). Per-class caption "a photo of a <name>."; box score =
1 - p(no-object); same SAM arbitration as probe_crossfam_owl.py.
"""
import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from torchvision import transforms

import data
from eval_seg import IoUMeter

THRESH = 0.7

NORM = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_mdetr(hub_dir, ckpt_path, device):
    sys.path.insert(0, f"{hub_dir}/ashkamath_mdetr_main")
    from hubconf import mdetr_efficientnetB5
    model, _ = mdetr_efficientnetB5(pretrained=False, return_postprocessor=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if "position_ids" not in k}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--n-gt-classes", type=int, required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--hub-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda"
    model = load_mdetr(a.hub_dir, a.ckpt, device)
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to(device)
    predictor = SamPredictor(sam)

    samples, _, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    names = json.load(open(a.vocab_file))
    Kv = len(names)
    labels = [n.split(",")[0].strip() for n in names]
    fg_ids = list(range(1, Kv))

    meter = IoUMeter(Kv, ignore)
    conf = np.zeros((a.n_gt_classes, Kv), dtype=np.int64)
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        w0, h0 = img.size
        t = NORM(img).unsqueeze(0).to(device)
        dets = []
        for cid in fg_ids:
            cap = f"a photo of a {labels[cid]}."
            with torch.no_grad():
                mem = model(t, [cap], encode_and_save=True)
                out = model(t, [cap], encode_and_save=False, memory_cache=mem)
            probs = 1 - out["pred_logits"].softmax(-1)[0, :, -1]
            keep = probs > THRESH
            if keep.any():
                bx = out["pred_boxes"][0, keep]
                cx, cy, bw, bh = bx.unbind(-1)
                boxes = torch.stack([(cx - bw / 2) * w0, (cy - bh / 2) * h0,
                                     (cx + bw / 2) * w0, (cy + bh / 2) * h0], -1)
                for b, s in zip(boxes.cpu().numpy(), probs[keep].cpu().numpy()):
                    dets.append((float(s), cid, b))
        score_map = np.zeros((h0, w0), dtype=np.float32)
        pred = np.zeros((h0, w0), dtype=np.int64)
        if dets:
            predictor.set_image(np.asarray(img))
            for s, cid, box in sorted(dets):
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                m = masks[0]
                take = m & (s > score_map)
                pred[take] = cid
                score_map[take] = s
        gta = np.asarray(gt)
        meter.update(pred, gta)
        mm = gta != ignore
        k = gta[mm].astype(np.int64) * Kv + pred[mm]
        conf += np.bincount(k, minlength=a.n_gt_classes * Kv)[:a.n_gt_classes * Kv] \
            .reshape(a.n_gt_classes, Kv)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}] mIoU={meter.miou()[0]*100:.2f}", flush=True)

    miou_gt, _ = meter.miou()
    np.savez_compressed(a.out + ".npz", conf=conf, kg=a.n_gt_classes, kv=Kv)
    json.dump({"prereg": "prereg_w7b_lineage.md", "model": "mdetr_efficientnetB5",
               "vocab": a.vocab_file, "miou_gt": miou_gt * 100,
               "miou_all": meter.miou_all() * 100}, open(a.out, "w"), indent=1)
    print(json.dumps({"vocab": a.vocab_file, "miou_gt": round(miou_gt * 100, 2),
                      "miou_all": round(meter.miou_all() * 100, 2)}))


if __name__ == "__main__":
    main()
