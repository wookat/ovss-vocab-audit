"""LPOSS-style and SC-CLIP-style dense inference (W3b style-reimplementations).

LPOSS (arXiv:2503.19777): DINO-affinity label propagation over patch-level
class logits of a kk-attention exit. SC-CLIP (arXiv:2411.15869): anomaly-token
restoration + kk attention, no residual/FFN.
"""
import torch
import torch.nn.functional as F
from clip_seg import DenseCLIP
from proxyclip_seg import CLIP_MEAN, CLIP_STD, IN_MEAN, IN_STD, ProxyCLIP


class SCCLIP(DenseCLIP):
    def __init__(self, device="cuda"):
        super().__init__(variant="naclip", device=device)

    def _last_block(self, blk, x):
        # anomaly-token restoration: patch tokens with pre-block norm > 5x median
        gh, gw = self._grid
        pat = x[:, 1:, :]
        norms = pat.norm(dim=-1)  # B,N
        med = norms.median(dim=1, keepdim=True).values
        bad = norms > 5.0 * med  # B,N
        if bad.any():
            B, N, C = pat.shape
            grid = pat.permute(0, 2, 1).reshape(B, C, gh, gw)
            neigh = F.avg_pool2d(grid, 3, stride=1, padding=1)
            neigh = neigh.reshape(B, C, N).permute(0, 2, 1)
            pat = torch.where(bad.unsqueeze(-1), neigh, pat)
            x = torch.cat([x[:, :1, :], pat], dim=1)
        attn = blk.attn
        ln_x = blk.ln_1(x)
        B, N, C = ln_x.shape
        qkv = ln_x @ attn.in_proj_weight.T + attn.in_proj_bias
        _, k, v = qkv.chunk(3, dim=-1)
        H = attn.num_heads
        d = C // H
        def split(t):
            return t.reshape(B, N, H, d).permute(0, 2, 1, 3)
        k, v = split(k), split(v)
        a = ((k @ k.transpose(-2, -1)) * d ** -0.5).softmax(-1)
        out = (a @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return attn.out_proj(out)  # no residual, no FFN


def dino_affinity(dino, img, knn):
    """Row-normalized kNN affinity over DINO ViT-S/16 patch features."""
    dev = img.device
    x = img * CLIP_STD.view(1, 3, 1, 1).to(dev) + CLIP_MEAN.view(1, 3, 1, 1).to(dev)
    x = (x - IN_MEAN.view(1, 3, 1, 1).to(dev)) / IN_STD.view(1, 3, 1, 1).to(dev)
    df = dino.get_intermediate_layers(x, n=1)[0][:, 1:]
    df = F.normalize(df.float(), dim=-1)
    S = df @ df.transpose(-2, -1)  # B,N,N
    N = S.shape[-1]
    kn = min(knn, N)
    topv, topi = S.topk(kn, dim=-1)
    Sk = torch.zeros_like(S).scatter_(-1, topi, topv.clamp_min(0))
    return Sk / Sk.sum(-1, keepdim=True).clamp_min(1e-8)


def propagate(Sk, feat, alpha, iters):
    Z0 = feat.float()
    Z = Z0
    for _ in range(iters):
        Z = alpha * (Sk @ Z) + (1 - alpha) * Z0
    return Z


class PropVariant(DenseCLIP):
    """W4c transplant probe: any surgery-family base + LPOSS-style propagation."""
    def __init__(self, variant, device="cuda", alpha=0.9, iters=10, knn=32):
        super().__init__(variant=variant, device=device)
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                                   skip_validation=True).to(device).eval()
        self.alpha, self.iters, self.knn = alpha, iters, knn

    @torch.no_grad()
    def encode_dense(self, img):
        feat, (gh, gw) = super().encode_dense(img)
        Sk = dino_affinity(self.dino, img, self.knn)
        return propagate(Sk, feat, self.alpha, self.iters), (gh, gw)


class PropSCCLIP(SCCLIP):
    """W4c transplant probe: SC-CLIP-style base + LPOSS-style propagation."""
    def __init__(self, device="cuda", alpha=0.9, iters=10, knn=32):
        super().__init__(device=device)
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                                   skip_validation=True).to(device).eval()
        self.alpha, self.iters, self.knn = alpha, iters, knn

    @torch.no_grad()
    def encode_dense(self, img):
        feat, (gh, gw) = super().encode_dense(img)
        Sk = dino_affinity(self.dino, img, self.knn)
        return propagate(Sk, feat, self.alpha, self.iters), (gh, gw)


class PropProxy(ProxyCLIP):
    """W4c transplant probe: ProxyCLIP-style base + LPOSS-style propagation."""
    def __init__(self, device="cuda", alpha=0.9, iters=10, knn=32):
        super().__init__(device=device)
        self.alpha_prop, self.iters_prop, self.knn_prop = alpha, iters, knn

    @torch.no_grad()
    def encode_dense(self, img):
        feat, (gh, gw) = super().encode_dense(img)
        Sk = dino_affinity(self.dino, img, self.knn_prop)
        return propagate(Sk, feat, self.alpha_prop, self.iters_prop), (gh, gw)


class LPOSS(DenseCLIP):
    """kk exit (ClearCLIP-style with k-k attention) + DINO label propagation.
    Propagation happens in embedding space per window: Z <- a*S Z + (1-a)*Z0."""
    def __init__(self, device="cuda", alpha=0.9, iters=10, knn=32):
        super().__init__(variant="clearclip", device=device)
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                                   skip_validation=True).to(device).eval()
        self.alpha, self.iters, self.knn = alpha, iters, knn

    def _last_block(self, blk, x):
        attn = blk.attn
        ln_x = blk.ln_1(x)
        B, N, C = ln_x.shape
        qkv = ln_x @ attn.in_proj_weight.T + attn.in_proj_bias
        _, k, v = qkv.chunk(3, dim=-1)
        H = attn.num_heads
        d = C // H
        def split(t):
            return t.reshape(B, N, H, d).permute(0, 2, 1, 3)
        ks, vs = split(k), split(v)
        a = ((ks @ ks.transpose(-2, -1)) * d ** -0.5).softmax(-1)
        out = (a @ vs).permute(0, 2, 1, 3).reshape(B, N, C)
        return attn.out_proj(out)

    @torch.no_grad()
    def encode_dense(self, img):
        feat, (gh, gw) = super().encode_dense(img)
        dev = img.device
        x = img * CLIP_STD.view(1, 3, 1, 1).to(dev) + CLIP_MEAN.view(1, 3, 1, 1).to(dev)
        x = (x - IN_MEAN.view(1, 3, 1, 1).to(dev)) / IN_STD.view(1, 3, 1, 1).to(dev)
        df = self.dino.get_intermediate_layers(x, n=1)[0][:, 1:]
        df = F.normalize(df.float(), dim=-1)
        S = df @ df.transpose(-2, -1)  # B,N,N
        # kNN sparsification + row normalization
        N = S.shape[-1]
        kn = min(self.knn, N)
        topv, topi = S.topk(kn, dim=-1)
        Sk = torch.zeros_like(S).scatter_(-1, topi, topv.clamp_min(0))
        Sk = Sk / Sk.sum(-1, keepdim=True).clamp_min(1e-8)
        Z0 = feat.float()
        Z = Z0
        for _ in range(self.iters):
            Z = self.alpha * (Sk @ Z) + (1 - self.alpha) * Z0
        return Z, (gh, gw)
