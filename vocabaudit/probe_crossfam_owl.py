"""W6-F2 second model (prereg_w6f2_crossfam.md): OWLv2 boxes + SAM masks.

Same harness conventions as probe_crossfam.py; OWLv2 takes independent text
queries so no vocabulary chunking is needed.
"""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                          OwlViTProcessor, OwlViTForObjectDetection)
from segment_anything import sam_model_registry, SamPredictor

import data
from eval_seg import IoUMeter

THRESH = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--n-gt-classes", type=int, required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda"
    if "owlvit" in a.model:
        proc = OwlViTProcessor.from_pretrained(a.model)
        owl = OwlViTForObjectDetection.from_pretrained(a.model).to(device).eval()
    else:
        proc = Owlv2Processor.from_pretrained(a.model)
        owl = Owlv2ForObjectDetection.from_pretrained(a.model).to(device).eval()
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to(device)
    predictor = SamPredictor(sam)

    samples, _, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    names = json.load(open(a.vocab_file))
    Kv = len(names)
    labels = [n.split(",")[0].strip() for n in names]
    fg_ids = list(range(1, Kv))
    queries = [labels[c] if a.bare else f"a photo of a {labels[c]}"
               for c in fg_ids]

    meter = IoUMeter(Kv, ignore)
    conf = np.zeros((a.n_gt_classes, Kv), dtype=np.int64)
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        w0, h0 = img.size
        inputs = proc(text=[queries], images=img, padding="max_length",
                      truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = owl(**inputs)
        det = proc.post_process_object_detection(
            out, threshold=THRESH, target_sizes=torch.tensor([(h0, w0)]))[0]
        score_map = np.zeros((h0, w0), dtype=np.float32)
        pred = np.zeros((h0, w0), dtype=np.int64)
        if len(det["boxes"]) > 0:
            predictor.set_image(np.asarray(img))
            order = torch.argsort(det["scores"])
            for j in order:
                cid = fg_ids[int(det["labels"][j])]
                box = det["boxes"][j].cpu().numpy()
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                m = masks[0]
                s = float(det["scores"][j])
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
    json.dump({"prereg": "prereg_w6f2_crossfam.md", "model": a.model,
               "vocab": a.vocab_file, "miou_gt": miou_gt * 100,
               "miou_all": meter.miou_all() * 100}, open(a.out, "w"), indent=1)
    print(json.dumps({"vocab": a.vocab_file, "miou_gt": round(miou_gt * 100, 2),
                      "miou_all": round(meter.miou_all() * 100, 2)}))


if __name__ == "__main__":
    main()
