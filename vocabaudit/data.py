"""Dataset sample lists + GT loaders. Roots on temp-hb: /media/dell/DATA/ovss/datasets."""
import os, glob
import numpy as np
from PIL import Image
import openseg_classes as oc

ROOT = os.environ.get("OVSS_DATA", "/media/dell/DATA/ovss/datasets")


def _png_loader(offset=0, ignore_zero=False):
    def load(p):
        g = np.asarray(Image.open(p)).astype(np.int64)
        if ignore_zero:
            g = g.copy(); g[g == 0] = 256; g = g - 1  # 0->255 ignore, k->k-1
            g[g == 255] = 255
        else:
            g = g + offset
        g[g > 60000] = 255
        return g
    return load


VOC21_NAMES = [  # SCLIP cls_voc21.txt convention: expanded background + synonym-rich person/tv
    "sky, wall, tree, wood, grass, road, sea, river, mountain, sands, desk, bed, building, cloud, lamp, door, window, wardrobe, ceiling, shelf, curtain, stair, floor, hill, rail, fence",
    "aeroplane", "bicycle", "bird", "ship", "bottle", "bus", "car", "cat", "chair", "cow",
    "table", "dog", "horse", "motorbike",
    "person, person in shirt, person in jeans, person in dress, person in sweater, person in skirt, person in jacket",
    "pottedplant", "sheep", "sofa", "train",
    "television monitor, tv monitor, monitor, television, screen",
]


def voc21():
    """VOC2012 val, 21 classes incl background(0)."""
    base = f"{ROOT}/VOCdevkit/VOC2012"
    ids = open(f"{base}/ImageSets/Segmentation/val.txt").read().split()
    samples = [(f"{base}/JPEGImages/{i}.jpg", f"{base}/SegmentationClass/{i}.png", _png_loader()) for i in ids]
    return samples, VOC21_NAMES, 255


def coco171(limit_list=None):
    """COCO-Stuff val2017: stuffthingmaps pngs carry raw ids (0..181, 255 ignore);
    remap to dense 0..170 via category id LUT."""
    lut = np.full(256, 255, dtype=np.int64)
    for i, c in enumerate(oc.COCO_STUFF_CATEGORIES):
        lut[c["id"] - 1] = i

    def load(p):
        g = np.asarray(Image.open(p)).astype(np.int64)
        return lut[g]

    imgs = sorted(glob.glob(f"{ROOT}/val2017/*.jpg"))
    samples = [(p, f"{ROOT}/cocostuff/val2017/{os.path.basename(p).replace('.jpg', '.png')}", load)
               for p in imgs]
    names = [c["name"] for c in oc.COCO_STUFF_CATEGORIES]
    return samples, names, 255


def ade150():
    base = f"{ROOT}/ADEChallengeData2016"
    imgs = sorted(glob.glob(f"{base}/images/validation/*.jpg"))
    samples = [(p, p.replace("images", "annotations").replace(".jpg", ".png"),
                _png_loader(ignore_zero=True)) for p in imgs]
    names = [c["name"] for c in oc.ADE20K_150_CATEGORIES]
    return samples, names, 255


def _mat_loader(p):
    from scipy.io import loadmat
    g = loadmat(p)["LabelMap"].astype(np.int64)
    g = g - 1          # 1..459 -> 0..458 ; 0 (unlabeled) -> -1
    g[g < 0] = 255
    return g


def pc459():
    """PASCAL-Context 459 full set: VOC2010 val images + stanford trainval mats."""
    base = f"{ROOT}/VOCdevkit/VOC2010"
    ids = open(f"{base}/ImageSets/Main/val.txt").read().split()
    mats = f"{ROOT}/pc_trainval"
    samples = []
    for i in ids:
        mp = f"{mats}/{i}.mat"
        if os.path.exists(mp):
            samples.append((f"{base}/JPEGImages/{i}.jpg", mp, _mat_loader))
    names = [c["name"] for c in oc.PASCAL_CTX_459_CATEGORIES]
    return samples, names, 255


def cocoobj():
    """COCO-Object style: 80 thing classes + background(0); stuff pixels -> background.
    Derived from COCO-Stuff maps: things are isthing=1 categories."""
    lut = np.full(256, 255, dtype=np.int64)
    things, names = [], ["background"]
    for c in oc.COCO_STUFF_CATEGORIES:
        if c.get("isthing", 0) == 1:
            things.append(c)
    for i, c in enumerate(things):
        lut[c["id"] - 1] = i + 1
        names.append(c["name"])
    for c in oc.COCO_STUFF_CATEGORIES:
        if c.get("isthing", 0) != 1:
            lut[c["id"] - 1] = 0  # stuff -> background

    def load(p):
        g = np.asarray(Image.open(p)).astype(np.int64)
        out = lut[g]
        out[g == 255] = 255
        return out

    imgs = sorted(glob.glob(f"{ROOT}/val2017/*.jpg"))
    samples = [(p, f"{ROOT}/cocostuff/val2017/{os.path.basename(p).replace('.jpg', '.png')}", load)
               for p in imgs]
    return samples, names, 255


def ctx60():
    """PASCAL-Context-60: 59 classes + background(0); other 459-classes -> background."""
    names459 = [c["name"] for c in oc.PASCAL_CTX_459_CATEGORIES]
    names59 = [c["name"] for c in oc.PASCAL_CTX_59_CATEGORIES]
    idx459 = {n: i for i, n in enumerate(names459)}
    idx459["diningtable"] = idx459["table"]  # 59-list naming variant
    lut = np.zeros(460, dtype=np.int64)  # default background
    names = ["background"]
    for j, n in enumerate(names59):
        lut[idx459[n]] = j + 1
        names.append(n)

    def load(p):
        from scipy.io import loadmat
        g = loadmat(p)["LabelMap"].astype(np.int64)  # 0 unlabeled, 1..459
        out = np.zeros_like(g)
        m = g > 0
        out[m] = lut[g[m] - 1]
        return out

    samples459 = pc459()[0]
    samples = [(ip, gp, load) for ip, gp, _ in samples459]
    return samples, names, 255


DATASETS = {"voc21": voc21, "coco171": coco171, "ade150": ade150, "pc459": pc459,
            "cocoobj": cocoobj, "ctx60": ctx60}
