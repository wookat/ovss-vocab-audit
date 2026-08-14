"""W13-L5 addendum: cross-dataset transfer of VOC-selected boost scalar."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import IoUMeter, class_embeddings, resize_short, to_tensor, \
    seg_logits

BSTAR = {"sclip": 0.03, "naclip": 0.05}
CELLS = [("ade150", "perturbed_vocabs/ade150_plain.json",
          "perturbed_vocabs/ade150_plain_vabs64.json"),
         ("ctx60", "perturbed_vocabs/ctx60_plain.json",
          "perturbed_vocabs/ctx60_plain_vabs64.json")]


@torch.no_grad()
def run(out_path, limit=300):
    res = {"prereg": "prereg_w13_l5_vabsboost.md#addendum"}
    for variant, bstar in BSTAR.items():
        model = DenseCLIP(variant, device="cuda")
        for ds, plain_f, vabs_f in CELLS:
            samples, _, ignore = data.DATASETS[ds]()
            names = json.load(open(plain_f))
            vnames = json.load(open(vabs_f))
            K = len(names)
            arms = {}
            for arm, nm in (("plain", names), ("vabs", vnames)):
                Kc = len(nm)
                # background channel: folded at 0 (ctx60) or appended (ade)
                if nm[0].startswith("background"):
                    bg_ch = 0
                elif Kc > K:
                    bg_ch = K
                else:
                    bg_ch = None  # ADE plain: no background to boost
                emb, qidx = class_embeddings(model, nm, "none")
                emb = emb.to(model.device)
                qidx = qidx.to(model.device)
                m0 = IoUMeter(Kc, ignore)
                mb = IoUMeter(Kc, ignore)
                for ip, gp, loader in samples[:limit]:
                    gt = loader(gp)
                    img = Image.open(ip).convert("RGB")
                    img_r, (w0, h0) = resize_short(img, 336)
                    t = to_tensor(img_r, model.device)
                    lg = seg_logits(model, t, emb, 224, 112)
                    lg = F.interpolate(lg.unsqueeze(0), size=(h0, w0),
                                       mode="bilinear",
                                       align_corners=False)[0]
                    if lg.shape[0] != Kc:
                        pooled = torch.full((Kc, *lg.shape[1:]), -1e4,
                                            device=lg.device)
                        idx = qidx.view(-1, 1, 1).expand_as(lg)
                        pooled.scatter_reduce_(0, idx, lg, reduce="amax",
                                               include_self=True)
                        lg = pooled
                    m0.update(lg.argmax(0).cpu().numpy().astype(np.int64),
                              gt)
                    if bg_ch is not None:
                        lg[bg_ch] += bstar
                        mb.update(
                            lg.argmax(0).cpu().numpy().astype(np.int64), gt)
                arms[arm] = m0.miou()[0] * 100
                arms[arm + "_boost"] = (mb.miou()[0] * 100
                                        if bg_ch is not None else None)
            res[f"{variant}/{ds}"] = arms
            print(variant, ds,
                  {k: (round(v, 2) if v is not None else None)
                   for k, v in arms.items()}, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    run(ap.parse_args().out)
