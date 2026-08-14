"""LexRO evaluation: plug adapted text embeddings (+ learned bg queries) into the
unified eval harness. Any base method, any vocabulary file."""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from lexro import TextAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="naclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--skip-dev", action="store_true")
    ap.add_argument("--no-adapter", action="store_true", help="frozen baseline")
    ap.add_argument("--no-bg", action="store_true", help="drop learned bg queries")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = DenseCLIP(a.variant)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    if a.skip_dev:
        samples = samples[:300] + samples[400:]
    K = len(plain_names)
    names = json.load(open(a.vocab))

    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    ck = torch.load(a.ckpt, map_location=model.device)
    if not a.no_adapter:
        adapter = TextAdapter().to(model.device)
        adapter.load_state_dict(ck["adapter"])
        adapter.eval()
        with torch.no_grad():
            T = adapter(T)
    if not a.no_bg:
        bgq = F.normalize(ck["bgq"].to(model.device), dim=-1)
        # attach bg queries to the background class (index of name containing 'background')
        bg_ci = next((i for i, n in enumerate(names)
                      if "background" in n.lower()), None)
        if bg_ci is None:
            bg_ci = K - 1 if "background" in plain_names[-1].lower() else 0
        T = torch.cat([T, bgq], 0)
        qi = torch.cat([qi, torch.full((bgq.shape[0],), bg_ci, dtype=qi.dtype)])

    meter = IoUMeter(K, ignore)
    scale = 40.0
    for i, (ip, gp, loader) in enumerate(samples):
        gt = loader(gp)
        img = Image.open(ip).convert("RGB")
        img_r, (w0, h0) = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        sims = seg_logits(model, t, T)
        sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                             align_corners=False)[0]
        probs = (scale * sims).softmax(0)
        pooled = torch.zeros(K, h0, w0, device=probs.device)
        idx = qi.to(probs.device).view(-1, 1, 1).expand_as(probs)
        pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
        meter.update(pooled.argmax(0).cpu().numpy(), gt)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}] {meter.miou()[0]*100:.2f}", flush=True)

    miou, per = meter.miou()
    res = {"ckpt": a.ckpt, "variant": a.variant, "dataset": a.dataset,
           "vocab": a.vocab, "skip_dev": bool(a.skip_dev),
           "no_adapter": bool(a.no_adapter), "no_bg": bool(a.no_bg),
           "miou": miou, "miou_all": meter.miou_all(), "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"miou": round(miou * 100, 2)}, indent=1))


if __name__ == "__main__":
    main()
