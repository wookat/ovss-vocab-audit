"""C4 evaluation: visual dense adapter inserted after encode_dense
(stage11_phase3/prereg_c6r2_c4.md)."""
import argparse, json
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, seg_logits, resize_short, to_tensor, IoUMeter
from c4_train import VisAdapter


class AdaptedCLIP(DenseCLIP):
    def __init__(self, variant, adapter, device="cuda"):
        super().__init__(variant=variant, device=device)
        self.adapter = adapter

    @torch.no_grad()
    def encode_dense(self, img):
        feat, (gh, gw) = super().encode_dense(img)
        if self.adapter is not None:
            feat = self.adapter(F.normalize(feat.float(), dim=-1), gh, gw)
        return feat, (gh, gw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="omit for frozen baseline")
    ap.add_argument("--variant", default="naclip")
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    adapter = None
    if a.ckpt:
        adapter = VisAdapter().cuda()
        adapter.load_state_dict(torch.load(a.ckpt, map_location="cuda")["adapter"])
        adapter.eval()
    model = AdaptedCLIP(a.variant, adapter)
    samples, plain_names, ignore = data.DATASETS[a.dataset]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)

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
           "vocab": a.vocab, "miou": miou, "miou_all": meter.miou_all(),
           "per_class": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"miou": round(miou * 100, 2)}, indent=1))


if __name__ == "__main__":
    main()
