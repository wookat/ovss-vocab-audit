"""Sliding-window evaluation + mIoU for training-free OVSS. Unified protocol:
input short side 336 (A-847/PC-459: 336), window 224, stride 112, logits mean-fused,
templates: fixed 20-template list identical everywhere. Background handled per dataset flag.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from clip_seg import DenseCLIP

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

TEMPLATES = [  # openai imagenet 80 (as used by SCLIP/ClearCLIP/NACLIP)
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
    'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
    'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.',
    'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.',
    'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.',
    'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.',
    'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.',
    'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.',
    'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
    'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.',
    'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]


def class_embeddings(model, names, whiten_mode="none", shrink=0.5, stats_emb=None):
    """names: list of class names; 'a,b' = sub-queries of one class (SCLIP convention).
    Returns (query_emb (Q,D), query_idx (Q,) mapping query->class).
    whiten_mode: none | center | zca ; stats computed on stats_emb (default: the query set)."""
    queries, query_idx = [], []
    for ci, n in enumerate(names):
        for s in [x.strip() for x in n.split(",")]:
            texts = [t.format(s) for t in TEMPLATES]
            e = model.encode_text_raw(texts)
            queries.append(F.normalize(e.mean(0), dim=-1))
            query_idx.append(ci)
    T = torch.stack(queries)  # (Q, D)
    idx = torch.tensor(query_idx)
    if whiten_mode == "none":
        return F.normalize(T, dim=-1), idx
    S = T if stats_emb is None else stats_emb
    mu = S.mean(0, keepdim=True)
    X = T - mu
    if whiten_mode == "center":
        return F.normalize(X, dim=-1), idx
    Xs = S - mu
    C = (Xs.T @ Xs) / max(Xs.shape[0] - 1, 1)
    C = (1 - shrink) * C + shrink * torch.eye(C.shape[0], device=C.device) * C.diagonal().mean()
    evals, evecs = torch.linalg.eigh(C)
    W = evecs @ torch.diag(evals.clamp_min(1e-8) ** -0.5) @ evecs.T
    return F.normalize(X @ W, dim=-1), idx


@torch.no_grad()
def seg_logits(model, img, text_emb, window=224, stride=112, logit_scale=1.0):
    """img: (1,3,H,W). Returns (K,H,W) logits."""
    _, _, H, W = img.shape
    K = text_emb.shape[0]
    out = torch.zeros(K, H, W, device=img.device)
    cnt = torch.zeros(1, H, W, device=img.device)
    hs = list(range(0, max(H - window, 0) + 1, stride))
    ws = list(range(0, max(W - window, 0) + 1, stride))
    if hs[-1] + window < H: hs.append(H - window)
    if ws[-1] + window < W: ws.append(W - window)
    for y in hs:
        for x in ws:
            crop = img[:, :, y:y+window, x:x+window]
            feat, (gh, gw) = model.encode_dense(crop)
            feat = F.normalize(feat.float(), dim=-1)
            logits = logit_scale * feat @ text_emb.T  # (1, N, K)
            logits = logits.reshape(1, gh, gw, K).permute(0, 3, 1, 2)
            logits = F.interpolate(logits, size=(window, window), mode="bilinear", align_corners=False)
            out[:, y:y+window, x:x+window] += logits[0]
            cnt[:, y:y+window, x:x+window] += 1
    return out / cnt


def resize_short(img_pil, short=336, max_long=2048):
    w, h = img_pil.size
    s = short / min(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    if max(nw, nh) > max_long:
        s2 = max_long / max(nw, nh)
        nw, nh = int(nw * s2), int(nh * s2)
    return img_pil.resize((nw, nh), Image.BILINEAR), (w, h)


def to_tensor(img_pil, device):
    a = torch.from_numpy(np.asarray(img_pil).copy()).float().div_(255.0)
    a = (a - torch.tensor(MEAN)) / torch.tensor(STD)
    return a.permute(2, 0, 1).unsqueeze(0).to(device)


class IoUMeter:
    def __init__(self, K, ignore=255):
        self.K, self.ignore = K, ignore
        self.inter = np.zeros(K); self.union = np.zeros(K); self.seen = np.zeros(K)
    def update(self, pred, gt):
        m = gt != self.ignore
        pred, gt = pred[m], gt[m]
        for c in np.unique(gt):
            self.seen[c] += 1
        k = (gt * self.K + pred)
        bc = np.bincount(k, minlength=self.K * self.K).reshape(self.K, self.K)
        self.inter += np.diag(bc)
        self.union += bc.sum(0) + bc.sum(1) - np.diag(bc)
    def miou(self):
        valid = self.seen > 0
        iou = self.inter[valid] / np.maximum(self.union[valid], 1)
        return float(iou.mean()), {int(i): float(v) for i, v in zip(np.where(valid)[0], iou)}
    def miou_all(self):
        """Standard mIoU over ALL K classes (classes never in GT and never predicted get IoU 0)."""
        iou = self.inter / np.maximum(self.union, 1)
        return float(iou.mean())


@torch.no_grad()
def evaluate(model, samples, class_names, whiten_mode="none", shrink=0.5, short=336,
             window=224, stride=112, limit=None, ignore=255, log_every=100, logit_scale=40.0,
             stats_names=None, offset=0):
    stats_emb = None
    if stats_names is not None:
        stats_emb, _ = class_embeddings(model, stats_names, "none")
    text_emb, query_idx = class_embeddings(model, class_names, whiten_mode, shrink, stats_emb=stats_emb)
    text_emb = text_emb.to(model.device)
    K = len(class_names)
    qidx = query_idx.to(model.device)
    pool_needed = text_emb.shape[0] != K
    meter = IoUMeter(K, ignore)
    subset = samples[offset:] if limit is None else samples[offset:offset + limit]
    for i, (img_path, gt_path, gt_loader) in enumerate(subset):
        img = Image.open(img_path).convert("RGB")
        img_r, (w0, h0) = resize_short(img, short)
        t = to_tensor(img_r, model.device)
        logits = seg_logits(model, t, text_emb, window, stride)  # (Q,H,W)
        logits = F.interpolate(logits.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False)[0]
        probs = (logit_scale * logits).softmax(0)  # (Q,H,W)
        if pool_needed:
            pooled = torch.zeros(K, *probs.shape[1:], device=probs.device)
            idx = qidx.view(-1, 1, 1).expand_as(probs)
            pooled.scatter_reduce_(0, idx, probs, reduce="amax", include_self=False)
            probs = pooled
        pred = probs.argmax(0).cpu().numpy().astype(np.int64)
        gt = gt_loader(gt_path)
        meter.update(pred, gt)
        if (i + 1) % log_every == 0:
            print(f"  [{i+1}] running mIoU={meter.miou()[0]*100:.2f}", flush=True)
    m, per_class = meter.miou()
    return m, per_class, meter.miou_all()
