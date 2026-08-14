import argparse, json, time, os
import torch
from clip_seg import DenseCLIP
from eval_seg import evaluate
import data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sclip",
                    choices=["maskclip", "sclip", "clearclip", "naclip", "proxyclip",
                             "lposs", "scclip"])
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--dataset", default="voc21", choices=list(data.DATASETS))
    ap.add_argument("--whiten", default="none", choices=["none", "center", "zca"])
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--short", type=int, default=336)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--skip-dev", action="store_true",
                    help="exclude images [300,400) of the split (dev-100)")
    ap.add_argument("--vocab-file", default=None, help="json list of class names overriding dataset names")
    ap.add_argument("--stats-vocab", default=None, help="json list of names for whitening statistics (global-stats baseline)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    samples, names, ignore = data.DATASETS[a.dataset]()
    if a.skip_dev:
        samples = samples[:300] + samples[400:]
    if a.vocab_file:
        names = json.load(open(a.vocab_file))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.variant == "proxyclip":
        from proxyclip_seg import ProxyCLIP
        model = ProxyCLIP(device=dev)
    elif a.variant == "lposs":
        from newgen_seg import LPOSS
        model = LPOSS(device=dev)
    elif a.variant == "scclip":
        from newgen_seg import SCCLIP
        model = SCCLIP(device=dev)
    else:
        model = DenseCLIP(a.variant, model_name=a.model, device=dev)
    t0 = time.time()
    stats_names = json.load(open(a.stats_vocab)) if a.stats_vocab else None
    miou, per_class, miou_all = evaluate(model, samples, names, a.whiten, a.shrink, a.short,
                                         limit=a.limit, ignore=ignore, stats_names=stats_names,
                                         offset=a.offset)
    res = dict(variant=a.variant, model=a.model, dataset=a.dataset, whiten=a.whiten, shrink=a.shrink,
               short=a.short, limit=a.limit, offset=a.offset, vocab_file=a.vocab_file,
               stats_vocab=a.stats_vocab, skip_dev=bool(a.skip_dev), miou=miou, miou_all=miou_all, secs=time.time() - t0)
    print(json.dumps(res))
    if a.out:
        res["per_class"] = per_class
        os.makedirs(os.path.dirname(a.out), exist_ok=True) if os.path.dirname(a.out) else None
        json.dump(res, open(a.out, "w"), indent=2)
