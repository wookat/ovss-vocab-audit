"""HeadLens E1 go/no-go (prereg_headlens_v1.md).

Exact per-head decomposition of the last CLIP block in output-only mode:
final_emb(S) = ln_post(sum_{h in S} out_h + b_out) @ proj  for head subset S.
Evaluates single heads, all-heads, oracle greedy subset, and label-free selection
for four attention flavors, on VOC dev-100 with the official vocabulary.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from clip_seg import DenseCLIP
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter

FLAVORS = ["vanilla", "qq", "kk", "ident"]


class HeadLens(DenseCLIP):
    def __init__(self, device="cuda"):
        super().__init__(variant="clearclip", device=device)

    @torch.no_grad()
    def head_outputs(self, img, flavor):
        """Returns per-head pre-out_proj contributions mapped through out_proj weight.
        out_h: (B, N, C) for each of H heads; plus out_proj bias. Also grid."""
        v = self.visual
        x = v.conv1(img)
        B, C, gh, gw = x.shape
        x = x.reshape(B, C, gh * gw).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._resized_pos(gh, gw).to(x.dtype)
        x = v.ln_pre(x)
        blocks = v.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        blk = blocks[-1]
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
            a = ((q @ k.transpose(-2, -1)) * scale).softmax(-1)
            out = a @ vv
        elif flavor == "qq":
            a = ((q @ q.transpose(-2, -1)) * scale).softmax(-1)
            out = a @ vv
        elif flavor == "kk":
            a = ((k @ k.transpose(-2, -1)) * scale).softmax(-1)
            out = a @ vv
        else:  # ident
            out = vv
        # per-head through out_proj: W (C,C) column blocks per head
        W = attn.out_proj.weight  # (C, C)
        outs = []
        for h in range(H):
            o = out[:, h]  # (B,N,d)
            Wh = W[:, h * d:(h + 1) * d]  # (C,d)
            outs.append(o @ Wh.T)  # (B,N,C)
        return torch.stack(outs, 0), attn.out_proj.bias, (gh, gw)

    @torch.no_grad()
    def subset_emb(self, outs, bias, subset):
        x = outs[list(subset)].sum(0) + bias
        x = self.visual.ln_post(x)
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x[:, 1:, :]  # drop CLS


def seg_pred(emb, gh, gw, T, qi, h0, w0, K, scale=40.0):
    feat = F.normalize(emb.float(), dim=-1)[0]  # (N,D)
    sims = feat @ T.T  # (N,Q)
    sims = sims.reshape(gh, gw, -1).permute(2, 0, 1)
    sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0), mode="bilinear",
                         align_corners=False)[0]
    probs = (scale * sims).softmax(0)
    pooled = torch.zeros(K, h0, w0, device=probs.device)
    idx = qi.to(probs.device).view(-1, 1, 1).expand_as(probs)
    pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
    return pooled.argmax(0).cpu().numpy(), probs


def margin_score(emb, T):
    """Label-free head quality: mean (top1 - top2) similarity margin over patches."""
    feat = F.normalize(emb.float(), dim=-1)[0]
    sims = feat @ T.T
    top2 = sims.topk(2, dim=-1).values
    return float((top2[:, 0] - top2[:, 1]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vocab", default="perturbed_vocabs/voc21_official.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = HeadLens()
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    Hn = 12

    res = {"prereg": "prereg_headlens_v1.md", "vocab": a.vocab,
           "dev": {"offset": a.offset, "n": a.limit}, "flavors": {}}

    for flavor in FLAVORS:
        # cache per-image head outputs is memory-heavy; recompute per arm instead:
        # evaluate all single heads + all-heads in one pass, collecting meters.
        meters = {f"h{h}": IoUMeter(K, ignore) for h in range(Hn)}
        meters["all"] = IoUMeter(K, ignore)
        margins = np.zeros(Hn)
        for i, (ip, gp, loader) in enumerate(samples):
            gt = loader(gp)
            img = Image.open(ip).convert("RGB")
            img_r, (w0, h0) = resize_short(img, 336)
            t = to_tensor(img_r, model.device)
            outs, bias, (gh, gw) = model.head_outputs(t, flavor)
            for h in range(Hn):
                emb = model.subset_emb(outs, bias, [h])
                pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
                meters[f"h{h}"].update(pred, gt)
                margins[h] += margin_score(emb, T)
            emb = model.subset_emb(outs, bias, range(Hn))
            pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
            meters["all"].update(pred, gt)
            if (i + 1) % 25 == 0:
                print(f"[{flavor} {i+1}] all={meters['all'].miou()[0]*100:.2f}", flush=True)
        single = [meters[f"h{h}"].miou()[0] * 100 for h in range(Hn)]
        margins /= len(samples)

        # oracle greedy subset (uses dev labels)
        chosen, best = [], -1
        remaining = list(range(Hn))
        while remaining:
            cand_best, cand_m = None, best
            for h in remaining:
                m = IoUMeter(K, ignore)
                for ip, gp, loader in samples[::4]:  # 1/4 subsample for greedy speed
                    gt = loader(gp)
                    img = Image.open(ip).convert("RGB")
                    img_r, (w0, h0) = resize_short(img, 336)
                    t = to_tensor(img_r, model.device)
                    outs, bias, (gh, gw) = model.head_outputs(t, flavor)
                    emb = model.subset_emb(outs, bias, chosen + [h])
                    pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
                    m.update(pred, gt)
                v = m.miou()[0] * 100
                if v > cand_m:
                    cand_m, cand_best = v, h
            if cand_best is None:
                break
            chosen.append(cand_best)
            remaining.remove(cand_best)
            best = cand_m
            print(f"[{flavor} greedy] +h{cand_best} -> {best:.2f}", flush=True)

        # evaluate oracle subset and label-free subset on FULL dev
        def full_eval(subset):
            m = IoUMeter(K, ignore)
            for ip, gp, loader in samples:
                gt = loader(gp)
                img = Image.open(ip).convert("RGB")
                img_r, (w0, h0) = resize_short(img, 336)
                t = to_tensor(img_r, model.device)
                outs, bias, (gh, gw) = model.head_outputs(t, flavor)
                emb = model.subset_emb(outs, bias, subset)
                pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
                m.update(pred, gt)
            return m.miou()[0] * 100

        oracle_miou = full_eval(chosen)
        order = np.argsort(-margins)
        lf_best_k, lf_best = 1, -1
        for kk in range(1, Hn + 1):
            # label-free k: pick k maximizing mean margin of the pooled subset on
            # first 10 images (no GT used)
            sub = order[:kk].tolist()
            s = 0.0
            for ip, gp, loader in samples[:10]:
                img = Image.open(ip).convert("RGB")
                img_r, (w0, h0) = resize_short(img, 336)
                t = to_tensor(img_r, model.device)
                outs, bias, (gh, gw) = model.head_outputs(t, flavor)
                s += margin_score(model.subset_emb(outs, bias, sub), T)
            if s > lf_best:
                lf_best, lf_best_k = s, kk
        lf_subset = order[:lf_best_k].tolist()
        lf_miou = full_eval(lf_subset)

        res["flavors"][flavor] = {
            "single_head_miou": single,
            "spread": max(single) - min(single),
            "all_heads_miou": meters["all"].miou()[0] * 100,
            "margin_scores": margins.tolist(),
            "oracle_subset": chosen, "oracle_miou": oracle_miou,
            "labelfree_subset": lf_subset, "labelfree_miou": lf_miou,
        }
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in res["flavors"][flavor].items()
                          if k != "single_head_miou"}, indent=1))

    # cross-flavor rank consistency
    from scipy.stats import spearmanr
    ranks = {f: res["flavors"][f]["single_head_miou"] for f in FLAVORS if f in res["flavors"]}
    cons = {}
    for f in FLAVORS[1:]:
        if f in ranks:
            cons[f"vanilla_vs_{f}"] = float(spearmanr(ranks["vanilla"], ranks[f]).statistic)
    res["rank_consistency"] = cons
    json.dump(res, open(a.out, "w"), indent=1)
    print("consistency:", cons)


if __name__ == "__main__":
    main()
