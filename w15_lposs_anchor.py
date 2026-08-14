"""W15 LPOSS author-code anchor: run unmodified LPOSS (CVPR'25 release) on VOC-21,
changing ONLY the class-name list (official / plain / syn100). Must be launched via
torchrun from the LPOSS repo root, e.g.:
  torchrun --nproc_per_node=1 w15_lposs_anchor.py --names plain --vocab-file .../voc21_plain.json
"""
import argparse
import json
import os

os.environ.setdefault("WANDB_MODE", "offline")

import wandb
from hydra import initialize, compose

import segmentation.datasets.pascal_voc as pv
from main_eval import main


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", choices=["official", "plain", "syn100"], required=True)
    ap.add_argument("--vocab-file", default=None,
                    help="JSON list of 21 names (background first); required for plain/syn100")
    ap.add_argument("--config", default="lposs.yaml")
    a = ap.parse_args()

    if a.names != "official":
        names = json.load(open(a.vocab_file))
        assert len(names) == 21 and names[0] == "background", names
        pv.PascalVOCDataset.CLASSES = tuple(names)
    print("CLASSES[0:3] =", pv.PascalVOCDataset.CLASSES[:3])

    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=a.config)
    wandb.init(project="LPOSS anchor", config=vars(a))
    main(cfg, "voc")


if __name__ == "__main__":
    cli()
