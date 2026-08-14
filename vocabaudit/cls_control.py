"""Classification-vs-segmentation sensitivity control (dense-specific evidence).

For each VOC-21 val image (foreground classes only), crop the tight bbox of each GT
class mask (pad 10%), classify the crop with the SAME query embeddings / pooling
convention as the segmentation evaluator, and report top-1 accuracy per vocab condition.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, MEAN, STD
import data


@torch.no_grad()
def classify_crop(model, img, emb, qidx, K):
    t = torch.from_numpy(np.asarray(img.resize((224, 224), Image.BILINEAR)).copy()).float().div_(255.0)
    t = ((t - torch.tensor(MEAN)) / torch.tensor(STD)).permute(2, 0, 1).unsqueeze(0).to(model.device)
    f = F.normalize(model.model.encode_image(t).float(), dim=-1)
    probs = (40.0 * f @ emb.T).softmax(-1)[0]
    pooled = torch.zeros(K, device=probs.device)
    pooled.scatter_reduce_(0, qidx.to(probs.device), probs, reduce="amax", include_self=False)
    return int(pooled.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-file", default=None)
    ap.add_argument("--whiten", default="none")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    samples, names, _ = data.voc21()
    if a.vocab_file:
        names = json.load(open(a.vocab_file))
    K = len(names)
    model = DenseCLIP("sclip", device="cuda" if torch.cuda.is_available() else "cpu")
    emb, qidx = class_embeddings(model, names, a.whiten)
    emb = emb.to(model.device)
    n_ok = n_tot = 0
    for img_path, gt_path, gt_loader in samples[: a.limit]:
        gt = gt_loader(gt_path)
        img = Image.open(img_path).convert("RGB")
        for c in np.unique(gt):
            if c in (0, 255):
                continue
            ys, xs = np.where(gt == c)
            if len(ys) < 400:
                continue
            h, w = gt.shape
            py, px = int(0.05 * (ys.max() - ys.min() + 1)), int(0.05 * (xs.max() - xs.min() + 1))
            box = (max(xs.min() - px, 0), max(ys.min() - py, 0),
                   min(xs.max() + px, w - 1) + 1, min(ys.max() + py, h - 1) + 1)
            pred = classify_crop(model, img.crop(box), emb, qidx, K)
            n_ok += int(pred == int(c))
            n_tot += 1
    res = dict(vocab_file=a.vocab_file, whiten=a.whiten, limit=a.limit,
               acc=n_ok / max(n_tot, 1), n=n_tot)
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
