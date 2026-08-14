"""C4: vocabulary-perturbation consistency training of a visual dense adapter
(stage11_phase3/prereg_c6r2_c4.md). Trains on the cached ADE NACLIP features."""
import argparse, glob, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from clip_seg import DenseCLIP
from lexro import eval_variant_names, build_variant_pool


class VisAdapter(nn.Module):
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.c1 = nn.Conv2d(dim, hidden, 1)
        self.c2 = nn.Conv2d(hidden, dim, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, f, gh=14, gw=14):
        """f: (B, P, D) normalized -> (B, P, D) normalized."""
        B, P, D = f.shape
        x = f.permute(0, 2, 1).reshape(B, D, gh, gw)
        r = self.c2(F.gelu(self.c1(x)))
        x = (x + r).reshape(B, D, P).permute(0, 2, 1)
        return F.normalize(x, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/media/dell/DATA/ovss/lexro_cache")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--nvar", type=int, default=4)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--w-drift", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    model = DenseCLIP("naclip")
    _, ade_names, _ = data.DATASETS["ade150"]()
    banned = eval_variant_names()
    pool = build_variant_pool(model, ade_names, banned)
    n_var = sum(len(p) for p in pool)
    print(f"variant pool: {n_var} names over {len(pool)} classes", flush=True)

    # embed every pool name once (template-averaged)
    from eval_seg import TEMPLATES
    emb_pool = []
    with torch.no_grad():
        for variants in pool:
            es = []
            for v in variants:
                e = model.encode_text_raw([t.format(v) for t in TEMPLATES])
                es.append(F.normalize(e.mean(0), dim=-1))
            emb_pool.append(torch.stack(es))
    T_canon = F.normalize(torch.stack([e[0] for e in emb_pool]), dim=-1).cuda()

    feats = []
    for sh in sorted(glob.glob(f"{a.cache}/shard_*.pt")):
        feats.append(torch.load(sh, map_location="cpu")["feat"])
    feats = torch.cat(feats)  # (N,196,512) f16 normalized
    N = feats.shape[0]
    print(f"cache: {N} images", flush=True)

    adapter = VisAdapter().cuda()
    opt = torch.optim.Adam(adapter.parameters(), lr=a.lr)
    scale = 40.0

    def sample_T():
        idx = [rng.randrange(len(e)) for e in emb_pool]
        return F.normalize(torch.stack(
            [e[i] for e, i in zip(emb_pool, idx)]), dim=-1).cuda()

    for ep in range(a.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        ent_sum, drift_sum = 0.0, 0.0
        for bi in range(0, N, a.batch):
            f = feats[perm[bi:bi + a.batch]].cuda().float()
            fa = adapter(f)
            Ts = [sample_T() for _ in range(a.nvar)]
            lps = [F.log_softmax(scale * fa @ T.T, -1).reshape(-1, len(pool))
                   for T in Ts]
            l_cons = 0.0
            npairs = 0
            for i in range(a.nvar):
                for j in range(i + 1, a.nvar):
                    l_cons = l_cons + 0.5 * (
                        F.kl_div(lps[i], lps[j], log_target=True,
                                 reduction="batchmean") +
                        F.kl_div(lps[j], lps[i], log_target=True,
                                 reduction="batchmean"))
                    npairs += 1
            l_cons = l_cons / npairs
            with torch.no_grad():
                p_frozen = F.log_softmax(scale * f @ T_canon.T, -1).reshape(-1, len(pool))
            lp_canon = F.log_softmax(scale * fa @ T_canon.T, -1).reshape(-1, len(pool))
            l_anchor = F.kl_div(lp_canon, p_frozen, log_target=True,
                                reduction="batchmean")
            l_drift = ((fa - f) ** 2).mean()
            loss = l_cons + a.w_anchor * l_anchor + a.w_drift * l_drift
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            with torch.no_grad():
                ent_sum += float(-(lp_canon.exp() * lp_canon).sum(-1).mean())
                drift_sum += float((fa - f).norm(dim=-1).mean())
        print(f"ep {ep+1} loss {tot/nb:.4f} entropy {ent_sum/nb:.3f} "
              f"drift {drift_sum/nb:.4f}", flush=True)
        if drift_sum / nb > 0.5:
            print("ABORT: drift collapse threshold exceeded", flush=True)
            break
        torch.save({"adapter": adapter.state_dict(), "epoch": ep + 1}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
