"""W5e step 2: offline flow decomposition (prereg_w5e_ba.md, frozen).

Given a plain-vocab confusion npz and a perturbed-vocab confusion npz for the
same (method, dataset), decompose the damage into:
- bg_absorption: extra GT-class pixel mass flowing to class 0 (background)
- steal: GT-class pixel mass flowing to extra vocab items (index >= Kg)
- interclass: extra GT-class pixel mass flowing to other GT classes
- metric_artifact: miou_gt - miou_all on the perturbed run (averaging over the
  enlarged / never-present class set)
Flows are reported as fractions of total GT pixel mass; mIoU terms in points.
"""
import argparse
import glob
import json
import os
import numpy as np


def flows(npz, has_bg=True):
    d = np.load(npz)
    conf = d["conf"].astype(np.float64)
    kg = int(d["kg"])
    tot = conf.sum()
    diag = sum(conf[c, c] for c in range(kg))
    bg = conf[1:, 0].sum() if has_bg else 0.0  # GT fg pixels predicted background
    steal = conf[:, kg:].sum() if conf.shape[1] > kg else 0.0
    inter = tot - diag - bg - steal
    return {"miou_gt": float(d["miou_gt"]) * 100,
            "miou_all": float(d["miou_all"]) * 100,
            "correct": diag / tot, "bg": bg / tot,
            "steal": steal / tot, "inter": inter / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--perturbed", required=True)
    ap.add_argument("--no-bg", action="store_true",
                    help="dataset has no background class (class 0 is semantic)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, q = flows(a.plain, not a.no_bg), flows(a.perturbed, not a.no_bg)
    res = {
        "plain": p, "perturbed": q,
        "delta_miou_all": q["miou_all"] - p["miou_all"],
        "delta_miou_gt": q["miou_gt"] - p["miou_gt"],
        "metric_artifact_pts": q["miou_gt"] - q["miou_all"],
        "d_bg": q["bg"] - p["bg"],
        "d_steal": q["steal"] - p["steal"],
        "d_inter": q["inter"] - p["inter"],
        "d_correct": q["correct"] - p["correct"],
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in res.items() if k not in ("plain", "perturbed")},
                     indent=1))


if __name__ == "__main__":
    main()
