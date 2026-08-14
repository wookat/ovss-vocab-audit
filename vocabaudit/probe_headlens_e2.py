"""HeadLens E2 (prereg_headlens_e2.md): label-free margin as predictor of surgery
config quality across flavor x exit-layer grid."""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from probe_headlens import HeadLens, seg_pred, margin_score, FLAVORS
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter

TOP3 = [3, 9, 11]  # frozen from E1 vanilla ranking


class HeadLensL(HeadLens):
    @torch.no_grad()
    def head_outputs_at(self, img, flavor, L):
        """Run blocks[:L-1], surgery attention at block index L-1 (1-based L)."""
        v = self.visual
        x = v.conv1(img)
        B, C, gh, gw = x.shape
        x = x.reshape(B, C, gh * gw).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._resized_pos(gh, gw).to(x.dtype)
        x = v.ln_pre(x)
        blocks = v.transformer.resblocks
        for blk in blocks[:L - 1]:
            x = blk(x)
        blk = blocks[L - 1]
        attn = blk.attn
        ln_x = blk.ln_1(x)
        B, N, C = ln_x.shape
        qkv = ln_x @ attn.in_proj_weight.T + attn.in_proj_bias
        q, k, vv = qkv.chunk(3, dim=-1)
        H = attn.num_heads
        d = C // H
        sp = lambda t: t.reshape(B, N, H, d).permute(0, 2, 1, 3)
        q, k, vv = sp(q), sp(k), sp(vv)
        scale = d ** -0.5
        if flavor == "vanilla":
            out = ((q @ k.transpose(-2, -1)) * scale).softmax(-1) @ vv
        elif flavor == "qq":
            out = ((q @ q.transpose(-2, -1)) * scale).softmax(-1) @ vv
        elif flavor == "kk":
            out = ((k @ k.transpose(-2, -1)) * scale).softmax(-1) @ vv
        else:
            out = vv
        W = attn.out_proj.weight
        outs = [out[:, h] @ W[:, h * d:(h + 1) * d].T for h in range(H)]
        return torch.stack(outs, 0), attn.out_proj.bias, (gh, gw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vocab", default="perturbed_vocabs/voc21_official.json")
    ap.add_argument("--layers", default="8,9,10,11,12")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = HeadLensL()
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    layers = [int(x) for x in a.layers.split(",")]
    Hn = 12

    res = {"prereg": "prereg_headlens_e2.md", "configs": {}}
    for L in layers:
        for flavor in FLAVORS:
            key = f"{flavor}_L{L}"
            m = IoUMeter(K, ignore)
            marg_full, marg_top3, n_marg = 0.0, 0.0, 0
            head_margins = np.zeros(Hn)
            for i, (ip, gp, loader) in enumerate(samples):
                gt = loader(gp)
                img = Image.open(ip).convert("RGB")
                img_r, (w0, h0) = resize_short(img, 336)
                t = to_tensor(img_r, model.device)
                outs, bias, (gh, gw) = model.head_outputs_at(t, flavor, L)
                emb = model.subset_emb(outs, bias, range(Hn))
                pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
                m.update(pred, gt)
                if i < 20:
                    marg_full += margin_score(emb, T)
                    marg_top3 += margin_score(model.subset_emb(outs, bias, TOP3), T)
                    for h in range(Hn):
                        head_margins[h] += margin_score(
                            model.subset_emb(outs, bias, [h]), T)
                    n_marg += 1
            res["configs"][key] = {
                "flavor": flavor, "layer": L,
                "miou": m.miou()[0] * 100,
                "margin_full": marg_full / n_marg,
                "margin_top3": marg_top3 / n_marg,
                "head_margins": (head_margins / n_marg).tolist(),
            }
            print(key, json.dumps({k: round(v, 3) for k, v in
                                   res["configs"][key].items()
                                   if isinstance(v, (int, float))}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)

    from scipy.stats import spearmanr
    cfgs = list(res["configs"].values())
    miou = [c["miou"] for c in cfgs]
    res["spearman_full"] = float(spearmanr(miou, [c["margin_full"] for c in cfgs]).statistic)
    res["spearman_top3"] = float(spearmanr(miou, [c["margin_top3"] for c in cfgs]).statistic)
    json.dump(res, open(a.out, "w"), indent=1)
    print("spearman full:", res["spearman_full"], "top3:", res["spearman_top3"])


if __name__ == "__main__":
    main()
