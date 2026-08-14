"""LexRO v2 (disclosed redesign, prereg_lexro_v1.md K-criteria unchanged):
pure text-side name normalization. Train the adapter to map every training-pool
variant embedding onto its class's canonical frozen embedding (cosine regression).
No images, no teacher. Held-out synonym vocabularies remain untouched."""
import argparse, json, random
import torch
import torch.nn.functional as F

from clip_seg import DenseCLIP
from lexro import TextAdapter, eval_variant_names, build_variant_pool
from lexro_train import embed_names
import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-anchor", type=float, default=1.0,
                    help="canonical names must stay fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda"

    model = DenseCLIP("naclip")
    coco_names = [n.split(",")[0].strip() for n in data.coco171()[1]]
    banned = eval_variant_names()
    pool = build_variant_pool(model, coco_names, banned)
    flat = [v for vs in pool for v in vs]
    owner, canon_idx, off = [], [], 0
    for ci, vs in enumerate(pool):
        canon_idx.append(off)
        owner += [ci] * len(vs)
        off += len(vs)
    owner = torch.tensor(owner, device=dev)
    canon_idx = torch.tensor(canon_idx, device=dev)
    E0 = embed_names(model, flat).to(dev)
    target = E0[canon_idx[owner]]          # canonical embedding per variant
    is_canon = torch.zeros(len(flat), dtype=torch.bool, device=dev)
    is_canon[canon_idx] = True
    print(f"{len(flat)} variants, {len(pool)} classes", flush=True)

    adapter = TextAdapter().to(dev)
    opt = torch.optim.AdamW(adapter.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        Ea = adapter(E0)
        L_norm = (1 - (Ea[~is_canon] * target[~is_canon]).sum(-1)).mean()
        L_anchor = (1 - (Ea[is_canon] * E0[is_canon]).sum(-1)).mean()
        loss = L_norm + a.w_anchor * L_anchor
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 200 == 0:
            print(f"ep{ep+1} norm={float(L_norm):.4f} anchor={float(L_anchor):.4f}",
                  flush=True)
    torch.save({"adapter": adapter.state_dict(),
                "bgq": torch.zeros(0, 512), "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
