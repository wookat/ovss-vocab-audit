"""LexRO stage B: train text adapter + bg queries on cached teacher shards
(prereg_lexro_v1.md). Vision/text towers frozen; only adapter + bg queries learn."""
import argparse, glob, json, math, random, time
import torch
import torch.nn.functional as F

from clip_seg import DenseCLIP
from eval_seg import TEMPLATES
from lexro import TextAdapter, eval_variant_names, build_variant_pool
import data


@torch.no_grad()
def embed_names(model, names):
    out = []
    for n in names:
        e = model.encode_text_raw([t.format(n) for t in TEMPLATES])
        out.append(F.normalize(e.mean(0), dim=-1))
    return torch.stack(out)


def effective_rank(E):
    s = torch.linalg.svdvals(E - E.mean(0))
    p = (s / s.sum()).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--vabs-meta", default="perturbed_vocabs/coco171_plain_vabs64_meta.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-inv", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--w-sep", type=float, default=1.0)
    ap.add_argument("--n-bg", type=int, default=16)
    ap.add_argument("--vocab-subsample", action="store_true",
                    help="per batch, keep a random subset of classes as foreground and"
                         " map all other teacher classes to background (run3 design)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda"

    model = DenseCLIP("naclip")  # text encoder only used here (frozen)
    coco_names = [n.split(",")[0].strip() for n in data.coco171()[1]]
    K = len(coco_names)  # 171; class K = background/teacher-negative
    banned = eval_variant_names()
    pool = build_variant_pool(model, coco_names, banned)
    n_var = sum(len(v) for v in pool)
    print(f"variant pool: {n_var} names for {K} classes "
          f"({sum(1 for v in pool if len(v) > 1)} classes with >=2 variants)", flush=True)

    # frozen embeddings for every variant
    flat = [v for vs in pool for v in vs]
    owner = torch.tensor([ci for ci, vs in enumerate(pool) for _ in vs], device=dev)
    E0 = embed_names(model, flat).to(dev)              # (V,512) frozen
    canon_idx = []
    off = 0
    for vs in pool:
        canon_idx.append(off); off += len(vs)
    canon_idx = torch.tensor(canon_idx, device=dev)
    frozen_canon = E0[canon_idx]
    frozen_offdiag = (frozen_canon @ frozen_canon.T - torch.eye(K, device=dev)).max()

    # bg queries init from VABS negatives
    negs = json.load(open(a.vabs_meta))["negatives"][:a.n_bg]
    bgq = torch.nn.Parameter(embed_names(model, negs).to(dev).clone())
    print("bg init:", negs, flush=True)

    adapter = TextAdapter().to(dev)
    opt = torch.optim.AdamW(list(adapter.parameters()) + [bgq], lr=a.lr)

    shards = sorted(glob.glob(f"{a.cache_dir}/shard_*.pt"))
    print(f"{len(shards)} shards", flush=True)
    scale = 40.0
    hist = []
    for ep in range(a.epochs):
        random.shuffle(shards)
        tot = {"distill": 0.0, "inv": 0.0, "anchor": 0.0, "sep": 0.0, "n": 0}
        for sp in shards:
            sh = torch.load(sp)
            feats, lbls, confs = sh["feat"], sh["lbl"], sh["conf"]
            idxs = list(range(feats.shape[0]))
            random.shuffle(idxs)
            for bi in range(0, len(idxs), a.bs):
                bidx = idxs[bi:bi + a.bs]
                f = feats[bidx].to(dev).float()          # (B,P,512)
                y = lbls[bidx].to(dev).long()            # (B,P) in [0..K]
                c = confs[bidx].to(dev).float()          # (B,P)

                Ea = adapter(E0)                          # (V,512)
                bq = F.normalize(bgq, dim=-1)             # (nbg,512)

                if a.vocab_subsample:
                    ksub = random.randint(10, 40)
                    present = y[y < K].unique().tolist()
                    random.shuffle(present)
                    sub = present[:ksub // 2]
                    rest = [ci for ci in range(K) if ci not in sub]
                    sub = sorted(sub + random.sample(rest, ksub - len(sub)))
                    cls_ids = sub
                    remap = torch.full((K + 1,), len(sub), dtype=torch.long, device=dev)
                    for j, ci in enumerate(sub):
                        remap[ci] = j
                    y_t = remap[y]
                    Kb = len(sub)
                else:
                    cls_ids = list(range(K))
                    y_t = y
                    Kb = K

                def logits_for(sel):
                    """sel: (Kb,) indices into flat variants. class Kb = max over bg queries."""
                    Wc = Ea[sel]                          # (K,512)
                    lc = scale * torch.einsum("bpd,kd->bpk", f, Wc)
                    lb = scale * torch.einsum("bpd,nd->bpn", f, bq).max(-1, keepdim=True).values
                    return torch.cat([lc, lb], dim=-1)    # (B,P,K+1)

                def sample_sel():
                    return torch.tensor(
                        [canon_idx[ci] if len(pool[ci]) == 1 else
                         canon_idx[ci] + random.randrange(len(pool[ci]))
                         for ci in cls_ids], device=dev)

                sel_canon = canon_idx[torch.tensor(cls_ids, device=dev)]
                l_canon = logits_for(sel_canon)
                L_distill = (F.cross_entropy(l_canon.reshape(-1, Kb + 1), y_t.reshape(-1),
                                             reduction="none") * c.reshape(-1)).mean()
                s1, s2 = sample_sel(), sample_sel()
                p1 = F.log_softmax(logits_for(s1), -1).reshape(-1, Kb + 1)
                p2 = F.log_softmax(logits_for(s2), -1).reshape(-1, Kb + 1)
                L_inv = 0.5 * (F.kl_div(p1, p2, log_target=True, reduction="batchmean")
                               + F.kl_div(p2, p1, log_target=True, reduction="batchmean"))
                L_anchor = (1 - (Ea * E0).sum(-1)).mean()
                adapted_canon = Ea[canon_idx]
                off_max = (adapted_canon @ adapted_canon.T - torch.eye(K, device=dev)).max()
                L_sep = F.relu(off_max - frozen_offdiag)

                loss = L_distill + a.w_inv * L_inv + a.w_anchor * L_anchor + a.w_sep * L_sep
                opt.zero_grad(); loss.backward(); opt.step()
                tot["distill"] += float(L_distill); tot["inv"] += float(L_inv)
                tot["anchor"] += float(L_anchor); tot["sep"] += float(L_sep); tot["n"] += 1
        with torch.no_grad():
            er = effective_rank(adapter(E0[canon_idx]))
        er0 = effective_rank(frozen_canon)
        n = max(tot["n"], 1)
        rec = {"epoch": ep, "distill": tot["distill"]/n, "inv": tot["inv"]/n,
               "anchor": tot["anchor"]/n, "sep": tot["sep"]/n,
               "eff_rank": er, "eff_rank_frozen": er0}
        hist.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in rec.items()}), flush=True)
        if er < 0.5 * er0:
            print("COLLAPSE ABORT", flush=True); break
        torch.save({"adapter": adapter.state_dict(), "bgq": bgq.detach().cpu(),
                    "args": vars(a), "hist": hist, "bg_words": negs},
                   a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
