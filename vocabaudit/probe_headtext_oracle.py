"""C2 HeadText oracle upper bound (stage11_phase3/prereg_c2c6_oracle.md).

Decomposed similarity sim(t,v) = sum_h w_h <P_h t, P_h v>, P_h = projector onto
head-h output subspace (linearised ln_post). Oracle: fit w on dev 1-50 with GT,
evaluate mIoU on dev 51-100 against the all-heads <t,v> baseline.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
from probe_headlens import HeadLens, seg_pred, FLAVORS
from eval_seg import class_embeddings, resize_short, to_tensor, IoUMeter


def head_projectors(model):
    """U_h (D, d) orthonormal bases of head output subspaces in final 512-d space."""
    v = model.visual
    attn = v.transformer.resblocks[-1].attn
    W = attn.out_proj.weight  # (C, C)
    g = v.ln_post.weight  # (C,)
    proj = v.proj  # (C, D)
    H = attn.num_heads
    d = W.shape[1] // H
    Us = []
    for h in range(H):
        Bh = (proj.T * g) @ W[:, h * d:(h + 1) * d]  # (D, d)
        U, _, _ = torch.linalg.svd(Bh.float(), full_matrices=False)
        Us.append(U)  # (D, d)
    return Us


@torch.no_grad()
def head_sims(feat, T, Us):
    """feat (N,D) normalized, T (Q,D) normalized -> (H, N, Q) projected sims."""
    sims = []
    for U in Us:
        sims.append((feat @ U) @ (T @ U).T)
    return torch.stack(sims, 0)


def pool_amax(logits, qi, K):
    """(N,Q) -> (N,K) amax over subqueries."""
    N = logits.shape[0]
    out = torch.full((N, K), -1e9, device=logits.device)
    idx = qi.to(logits.device).view(1, -1).expand(N, -1)
    out.scatter_reduce_(1, idx, logits, reduce="amax", include_self=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--vocab", default="perturbed_vocabs/voc21_official.json")
    ap.add_argument("--flavors", default="vanilla,qq")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = HeadLens()
    samples, plain_names, ignore = data.DATASETS["voc21"]()
    samples = samples[a.offset:a.offset + a.limit]
    K = len(plain_names)
    names = json.load(open(a.vocab))
    T, qi = class_embeddings(model, names)
    T = T.to(model.device)
    Us = head_projectors(model)
    Hn = len(Us)
    train_s, test_s = samples[:50], samples[50:]

    res = {"prereg": "stage11_phase3/prereg_c2c6_oracle.md", "flavors": {}}
    for flavor in a.flavors.split(","):
        assert flavor in FLAVORS
        # ---- cache train features ----
        feats, gts = [], []
        for ip, gp, loader in train_s:
            gt = loader(gp)
            img = Image.open(ip).convert("RGB")
            img_r, (w0, h0) = resize_short(img, 336)
            t = to_tensor(img_r, model.device)
            outs, bias, (gh, gw) = model.head_outputs(t, flavor)
            emb = model.subset_emb(outs, bias, range(Hn))
            feat = F.normalize(emb.float(), dim=-1)[0]  # (N,D)
            g = torch.from_numpy(np.array(
                Image.fromarray(gt.astype(np.int32), mode="I").resize(
                    (gw, gh), Image.NEAREST))).reshape(-1)
            feats.append(feat.cpu())
            gts.append(g)
        S_all, y_all = [], []
        for feat, g in zip(feats, gts):
            S = head_sims(feat.to(model.device), T, Us)  # (H,N,Q)
            keep = (g != ignore) & (g >= 0) & (g < K)
            S_all.append(S[:, keep.to(model.device) if S.is_cuda else keep].cpu())
            y_all.append(g[keep])
        S_tr = torch.cat(S_all, 1).to(model.device)  # (H, M, Q)
        y_tr = torch.cat(y_all).to(model.device).long()
        print(f"[{flavor}] train patches: {y_tr.shape[0]}", flush=True)

        # ---- fit w ----
        w = torch.zeros(Hn, device=model.device, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for step in range(300):
            wt = F.softplus(w) * Hn / F.softplus(w).sum().detach().clamp_min(1e-6)
            logits = torch.einsum("h,hmq->mq", wt, S_tr)
            cls = pool_amax(40.0 * logits, qi, K)
            loss = F.cross_entropy(cls, y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 100 == 0:
                print(f"[{flavor}] step {step+1} loss {loss.item():.4f}", flush=True)
        wt = (F.softplus(w) * Hn / F.softplus(w).sum().clamp_min(1e-6)).detach()

        # ---- eval on held 50 ----
        def eval_arms(weights):
            m_dec = IoUMeter(K, ignore)
            m_base = IoUMeter(K, ignore)
            for ip, gp, loader in test_s:
                gt = loader(gp)
                img = Image.open(ip).convert("RGB")
                img_r, (w0, h0) = resize_short(img, 336)
                t = to_tensor(img_r, model.device)
                outs, bias, (gh, gw) = model.head_outputs(t, flavor)
                emb = model.subset_emb(outs, bias, range(Hn))
                feat = F.normalize(emb.float(), dim=-1)[0]
                # baseline
                pred, _ = seg_pred(emb, gh, gw, T, qi, h0, w0, K)
                m_base.update(pred, gt)
                # decomposed
                S = head_sims(feat, T, Us)
                sims = torch.einsum("h,hnq->nq", weights, S)
                sims = sims.reshape(gh, gw, -1).permute(2, 0, 1)
                sims = F.interpolate(sims.unsqueeze(0), size=(h0, w0),
                                     mode="bilinear", align_corners=False)[0]
                probs = (40.0 * sims).softmax(0)
                pooled = torch.zeros(K, h0, w0, device=probs.device)
                idx = qi.to(probs.device).view(-1, 1, 1).expand_as(probs)
                pooled.scatter_reduce_(0, idx, probs, reduce="amax",
                                       include_self=False)
                m_dec.update(pooled.argmax(0).cpu().numpy(), gt)
            return m_dec.miou()[0] * 100, m_base.miou()[0] * 100

        dec, base = eval_arms(wt)
        uni, _ = eval_arms(torch.ones(Hn, device=model.device))
        res["flavors"][flavor] = {
            "w": wt.cpu().tolist(),
            "oracle_decomposed_miou": dec,
            "uniform_decomposed_miou": uni,
            "allheads_baseline_miou": base,
            "gain": dec - base,
        }
        print(flavor, json.dumps({k: (round(v, 3) if isinstance(v, float) else
                                      [round(x, 3) for x in v])
                                  for k, v in res["flavors"][flavor].items()}),
              flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
