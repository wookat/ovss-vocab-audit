"""W6-F2 (prereg_w6f2_crossfam.md, frozen): naming-perturbation audit of a
grounding-family segmenter (Grounding DINO boxes + SAM box-prompted masks).

Per image: one grounded-detection pass over the full vocabulary text, SAM
mask per box, per-pixel argmax by detection score; unclaimed pixels ->
background (class 0). Same IoUMeter conventions as the main pipeline.
"""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

import data
from eval_seg import IoUMeter

BOX_THRESH = 0.25
TEXT_THRESH = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--n-gt-classes", type=int, required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    gdino = GroundingDinoForObjectDetection.from_pretrained(a.model).to(device).eval()
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to(device)
    predictor = SamPredictor(sam)

    samples, _, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    names = json.load(open(a.vocab_file))
    Kv = len(names)
    # first sub-query per class; skip class 0 (background) as a query
    labels = [n.split(",")[0].strip() for n in names]
    fg_ids = list(range(1, Kv))
    # BERT text tower caps at 256 tokens: chunk the vocabulary
    CHUNK = 40
    chunks = [fg_ids[i:i + CHUNK] for i in range(0, len(fg_ids), CHUNK)]
    texts = [". ".join(labels[c] for c in ch) + "." for ch in chunks]

    meter = IoUMeter(Kv, ignore)
    conf = np.zeros((a.n_gt_classes, Kv), dtype=np.int64)
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        w0, h0 = img.size
        dets = []  # (score, box, class-id)
        lab2id = {labels[c].lower(): c for c in fg_ids}
        for text in texts:
            inputs = proc(images=img, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = gdino(**inputs)
            try:
                det = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=BOX_THRESH,
                    text_threshold=TEXT_THRESH, target_sizes=[(h0, w0)])[0]
            except TypeError:
                det = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=BOX_THRESH,
                    text_threshold=TEXT_THRESH, target_sizes=[(h0, w0)])[0]
            for j in range(len(det["boxes"])):
                lab = det["text_labels"][j] if "text_labels" in det else det["labels"][j]
                cid = lab2id.get(str(lab).lower().strip())
                if cid is None:
                    for k, c in lab2id.items():
                        if str(lab).lower().strip() in k or k in str(lab).lower():
                            cid = c
                            break
                if cid is None:
                    continue
                dets.append((float(det["scores"][j]),
                             det["boxes"][j].cpu().numpy(), cid))
        score_map = np.zeros((h0, w0), dtype=np.float32)
        pred = np.zeros((h0, w0), dtype=np.int64)  # background
        if dets:
            predictor.set_image(np.asarray(img))
            for s, box, cid in sorted(dets, key=lambda d: d[0]):
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
    json.dump({"prereg": "prereg_w6f2_crossfam.md", "vocab": a.vocab_file,
               "miou_gt": miou_gt * 100, "miou_all": meter.miou_all() * 100},
              open(a.out, "w"), indent=1)
    print(json.dumps({"vocab": a.vocab_file, "miou_gt": round(miou_gt * 100, 2),
                      "miou_all": round(meter.miou_all() * 100, 2)}))


if __name__ == "__main__":
    main()
