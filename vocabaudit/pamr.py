"""PAMR: Pixel-Adaptive Mask Refinement (Araslanov & Roth, CVPR 2020).

Standard implementation as used by TCL/NACLIP: iteratively refines class
probability maps with an affinity kernel computed from the input image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAffinity(nn.Module):
    def __init__(self, dilations=(1,)):
        super().__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer("kernel", weight)

    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)
        idx = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                weight[idx, 0, 1 + dy, 1 + dx] = 1.0
                idx += 1
        return weight

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, 8*len(dilations), H, W) neighbour values
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        outs = []
        for d in self.dilations:
            o = F.conv2d(F.pad(x, (d,) * 4, mode="replicate"), self.kernel, dilation=d)
            outs.append(o)
        out = torch.cat(outs, dim=1)
        return out.reshape(B, C, -1, H, W)


class PAMR(nn.Module):
    def __init__(self, num_iter=10, dilations=(1, 2, 4, 8, 12, 24)):
        super().__init__()
        self.num_iter = num_iter
        self.aff = LocalAffinity(dilations)

    def forward(self, img, masks):
        """img (B,3,H,W) in [0,1]-ish; masks (B,K,H,W) probabilities."""
        img = F.interpolate(img, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        # affinity from image: std-normalised abs difference
        x = self.aff(img)                              # (B,3,N,H,W)
        x = (x - img.unsqueeze(2)).abs()               # difference to centre
        x = x.mean(1, keepdim=True)                    # (B,1,N,H,W)
        x = -x / (1e-8 + 0.1 * x.std(dim=(2, 3, 4), keepdim=True))
        aff = x.softmax(2)                             # (B,1,N,H,W)
        for _ in range(self.num_iter):
            m = self.aff(masks)                        # (B,K,N,H,W)
            masks = (aff * m).sum(2)
        return masks
