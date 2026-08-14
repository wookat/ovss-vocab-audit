"""ProxyCLIP-style dense inference (arXiv:2408.04883) in the unified pipeline.

Proxy attention: DINO patch-affinity S (cosine), adaptively normalized
(S - mean)*gamma with sub-mean entries masked, softmax, applied to CLIP
last-block VALUE features (MaskCLIP pathway). DINO ViT-S/16 stand-in for the
paper's ViT-B/8; disclosed as style-reimplementation.
"""
import torch
import torch.nn.functional as F
from clip_seg import DenseCLIP

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IN_MEAN = torch.tensor([0.485, 0.456, 0.406])
IN_STD = torch.tensor([0.229, 0.224, 0.225])


class ProxyCLIP(DenseCLIP):
    def __init__(self, device="cuda", gamma=3.0):
        super().__init__(variant="maskclip", device=device)
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                                   skip_validation=True).to(device).eval()
        self.gamma = gamma

    @torch.no_grad()
    def _dino_affinity(self, img):
        # img is CLIP-normalized; convert to ImageNet normalization
        dev = img.device
        x = img * CLIP_STD.view(1, 3, 1, 1).to(dev) + CLIP_MEAN.view(1, 3, 1, 1).to(dev)
        x = (x - IN_MEAN.view(1, 3, 1, 1).to(dev)) / IN_STD.view(1, 3, 1, 1).to(dev)
        f = self.dino.get_intermediate_layers(x, n=1)[0][:, 1:]  # B,N,C
        f = F.normalize(f.float(), dim=-1)
        return f @ f.transpose(-2, -1)  # B,N,N

    @torch.no_grad()
    def encode_dense(self, img):
        self._proxy_S = self._dino_affinity(img)
        return super().encode_dense(img)

    def _last_block(self, blk, x):
        attn = blk.attn
        ln_x = blk.ln_1(x)
        B, N, C = ln_x.shape
        qkv = ln_x @ attn.in_proj_weight.T + attn.in_proj_bias
        _, _, v = qkv.chunk(3, dim=-1)
        S = self._proxy_S  # B, N-1, N-1 (patches only)
        S = (S - S.mean(dim=(-2, -1), keepdim=True)) * self.gamma
        S = S.masked_fill(S < 0, float("-inf"))
        P = S.softmax(-1).to(v.dtype)
        out = v.clone()
        out[:, 1:, :] = P @ v[:, 1:, :]
        return attn.out_proj(out)  # no residual, no FFN (ClearCLIP-style exit)
