"""Unified training-free OVSS inference for CLIP ViT: MaskCLIP / SCLIP / ClearCLIP.

Single implementation, one protocol: last-layer attention surgery on open_clip ViT-B/16.
- maskclip: value-only pathway (attn = identity on v), keep residual+FFN off per ClearCLIP? (orig MaskCLIP keeps them)
- sclip:    correlative self-attention  Attn(q,q)+Attn(k,k), keep residual & FFN (per SCLIP paper)
- clearclip: q-q attention, drop residual connection and FFN at last block
All return patch-level features projected to CLIP embedding space, L2-normalized.
"""
import torch
import torch.nn.functional as F
import open_clip


class DenseCLIP(torch.nn.Module):
    def __init__(self, variant="sclip", model_name="ViT-B-16-quickgelu", pretrained="openai", device="cuda"):
        super().__init__()
        assert variant in ("maskclip", "sclip", "clearclip", "naclip")
        self.variant = variant
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(device)
        self.device = device
        self.visual = self.model.visual
        self.patch_size = self.visual.conv1.kernel_size[0]

    @torch.no_grad()
    def encode_text_raw(self, texts, batch=256):
        embs = []
        for i in range(0, len(texts), batch):
            t = self.tokenizer(texts[i:i+batch]).to(self.device)
            e = self.model.encode_text(t)
            embs.append(F.normalize(e.float(), dim=-1))
        return torch.cat(embs)

    @torch.no_grad()
    def encode_dense(self, img):
        """img: (B,3,H,W) normalized. Returns (B, N, D) patch embeddings (unnormalized proj)."""
        v = self.visual
        x = v.conv1(img)
        B, C, gh, gw = x.shape
        x = x.reshape(B, C, gh * gw).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = self._resized_pos(gh, gw).to(x.dtype)
        x = x + pos
        x = v.patch_dropout(x) if hasattr(v, "patch_dropout") else x
        x = v.ln_pre(x)
        blocks = v.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        self._grid = (gh, gw)
        x = self._last_block(blocks[-1], x)
        x = v.ln_post(x)
        if v.proj is not None:
            x = x @ v.proj
        return x[:, 1:, :], (gh, gw)

    def _resized_pos(self, gh, gw):
        pos = self.visual.positional_embedding  # (1+N0, C)
        cls_pos, patch_pos = pos[:1], pos[1:]
        n0 = int(patch_pos.shape[0] ** 0.5)
        if (gh, gw) == (n0, n0):
            return pos.unsqueeze(0)
        p = patch_pos.reshape(1, n0, n0, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(gh, gw), mode="bicubic", align_corners=False)
        p = p.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return torch.cat([cls_pos.unsqueeze(0), p], dim=1)

    def _last_block(self, blk, x):
        attn = blk.attn
        ln_x = blk.ln_1(x)
        B, N, C = ln_x.shape
        w = attn.in_proj_weight
        b = attn.in_proj_bias
        qkv = ln_x @ w.T + b
        q, k, v = qkv.chunk(3, dim=-1)
        H = attn.num_heads
        d = C // H
        def split(t):
            return t.reshape(B, N, H, d).permute(0, 2, 1, 3)  # B,H,N,d
        q, k, v = split(q), split(k), split(v)
        scale = d ** -0.5
        if self.variant == "maskclip":
            out = v
        elif self.variant == "sclip":
            a = ((q @ q.transpose(-2, -1)) * scale).softmax(-1) + ((k @ k.transpose(-2, -1)) * scale).softmax(-1)
            out = a @ v
        elif self.variant == "naclip":
            a = (k @ k.transpose(-2, -1)) * scale + self._gauss_bias(N).to(k.dtype)
            out = a.softmax(-1) @ v
        else:  # clearclip
            a = (q @ q.transpose(-2, -1)) * scale
            out = a.softmax(-1) @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = attn.out_proj(out)
        if self.variant in ("clearclip", "naclip"):
            return out  # no residual, no FFN
        x = x + out
        x = x + blk.mlp(blk.ln_2(x))
        return x

    def _gauss_bias(self, N, std=5.0):
        """NACLIP neighbourhood prior: additive Gaussian attention bias on patch grid."""
        gh, gw = self._grid
        key = (gh, gw, N)
        if getattr(self, "_gb_key", None) == key:
            return self._gb
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        coords = torch.stack([ys.flatten(), xs.flatten()], dim=1).float()  # (P,2)
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        g = -d2 / (2 * std ** 2)
        bias = torch.zeros(N, N)
        bias[1:, 1:] = g
        self._gb_key, self._gb = key, bias.to(self.device)
        return self._gb
