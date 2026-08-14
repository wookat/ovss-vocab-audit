"""W10-H3 (prereg_w10_h3_recipe.md, frozen): MM-Grounding-DINO recipe ladder
under the same box->SAM harness as F2/W7b."""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from mmdet.apis import DetInferencer
from segment_anything import sam_model_registry, SamPredictor

import data
from eval_seg import IoUMeter

BOX_THRESH = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--n-gt-classes", type=int, required=True)
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    inf = DetInferencer(model=a.config, weights=a.ckpt, device="cuda",
                        show_progress=False)
    sam = sam_model_registry["vit_b"](checkpoint=a.sam_ckpt).to("cuda")
    predictor = SamPredictor(sam)

    samples, _, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    names = json.load(open(a.vocab_file))
    Kv = len(names)
    labels = [n.split(",")[0].strip() for n in names]
    fg_ids = list(range(1, Kv))
    text = " . ".join(labels[c] for c in fg_ids) + " ."

    meter = IoUMeter(Kv, ignore)
    conf = np.zeros((a.n_gt_classes, Kv), dtype=np.int64)
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        res = inf(inputs=ip, texts=text, custom_entities=True,
                  return_datasamples=False, no_save_vis=True,
                  pred_score_thr=BOX_THRESH)["predictions"][0]
        dets = []
        for s, box, li in zip(res["scores"], res["bboxes"], res["labels"]):
            if s < BOX_THRESH:
                continue
            dets.append((float(s), np.array(box, dtype=np.float32),
                         fg_ids[int(li)]))
        img = Image.open(ip).convert("RGB")
        w0, h0 = img.size
        score_map = np.zeros((h0, w0), dtype=np.float32)
        pred = np.zeros((h0, w0), dtype=np.int64)
        if dets:
            predictor.set_image(np.asarray(img))
            for s, box, cid in sorted(dets, key=lambda d: d[0]):
                masks, _, _ = predictor.predict(box=box,
                                                multimask_output=False)
                m = masks[0]
                take = m & (s > score_map)
                pred[take] = cid
                score_map[take] = s
        gta = np.asarray(gt)
        meter.update(pred, gta)
        mm = gta != ignore
        k = gta[mm].astype(np.int64) * Kv + pred[mm]
        conf += np.bincount(
            k, minlength=a.n_gt_classes * Kv)[:a.n_gt_classes * Kv] \
            .reshape(a.n_gt_classes, Kv)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}] mIoU={meter.miou()[0]*100:.2f}", flush=True)

    miou_gt, _ = meter.miou()
    np.savez_compressed(a.out + ".npz", conf=conf, kg=a.n_gt_classes, kv=Kv)
    json.dump({"prereg": "prereg_w10_h3_recipe.md", "vocab": a.vocab_file,
               "ckpt": os.path.basename(a.ckpt),
               "miou_gt": miou_gt * 100,
               "miou_all": meter.miou_all() * 100}, open(a.out, "w"),
              indent=1)
    print(json.dumps({"ckpt": os.path.basename(a.ckpt),
                      "vocab": a.vocab_file,
                      "miou_gt": round(miou_gt * 100, 2)}))


if __name__ == "__main__":
    main()
