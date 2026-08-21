# Vocabulary-Robustness Audit + REVA — reproducibility artifact

Anonymous artifact accompanying:
1. *How Fragile Is Your Vocabulary? A Controlled Audit of
   Inference-Vocabulary Robustness in Training-free OVSS* (audit paper)
2. *REVA: Vocabulary-Adaptive Background Synthesis and Region-Evidence
   Arbitration for Plain-Vocabulary OVSS* (method paper)

## Contents
- `vocabaudit/` — the complete unified pipeline: dense CLIP inference
  (`clip_seg.py`, `eval_seg.py`, `data.py`), all perturbation
  vocabularies (`perturbed_vocabs/`), and every experiment runner
  (`probe_*.py`, one file per pre-registered experiment; the file name
  encodes the week/index used in the papers and in `results/`).
- `preregs/` — all pre-registration files. Each freezes hypothesis,
  protocol, and GO/NO-GO/kill criteria *before* the corresponding run.
- `results/` — the verdict log (`RESULTS_*.md`): every pre-registered
  experiment with its frozen criterion and honest verdict
  (GO / NO-GO / MIXED / infeasible), including all negative results
  indexed in the audit paper's kill-table appendix.
- `paper_audit/`, `paper_reva/` — LaTeX sources.

## Protocol (both papers)
OpenCLIP (OpenAI weights), ViT-B/16 default; short side 336, sliding
window 224 / stride 112; logit scale 40. SAM ViT-B
(`sam_vit_b_01ec64.pth`), points_per_side 16. OWLv2
`google/owlv2-base-patch16-ensemble`, box threshold 0.2 (frozen, untuned).

## Environment
`pip install -r requirements.txt` (verified: Python 3.10,
torch 2.4.1+cu121, single RTX 3090 24 GB). NLTK WordNet:
`python -c "import nltk; nltk.download('wordnet')"`. Weights:
OpenCLIP/OWLv2 auto-download; SAM `sam_vit_b_01ec64.pth` from the
official Meta release. Official-repo anchor reproductions
(ProxyCLIP/LPOSS/SC-CLIP/Trident/RF-CLIP/CorrCLIP/SCI-CLIP) each used
that repo's own environment, recorded per anchor in
`results/RESULTS_*.md`.

## Datasets
Point `OVSS_DATA` (default `/media/dell/DATA/ovss/datasets`) at a
directory containing the standard layouts expected by
`vocabaudit/data.py`:
- `VOCdevkit/VOC2012` and `VOCdevkit/VOC2010` (PASCAL VOC / Context;
  Context .mat annotations in `pc_trainval/`)
- `val2017/` + `cocostuff/val2017/` (COCO images + COCO-Stuff PNG masks)
- `ADEChallengeData2016/`
No dataset images or annotations are redistributed in this artifact.

## Reproducing
Each `probe_*.py` is standalone:
```
OVSS_DATA=/path/to/datasets python probe_w11_j5_prune.py --out runs/j5.json
```
Datasets (VOC-21, Context-60, COCO-Stuff-171, COCO-Object, ADE-150,
A-847, PC-459) are loaded via `data.py`.

To rebuild the papers: `pdflatex main.tex` twice in `paper_audit/` or
`paper_reva/` (bibliography is inline `thebibliography`; no bibtex run
needed).

## Continuing this work
The intended edit loop is: (1) write a new `preregs/prereg_*.md`
freezing hypothesis + criteria, (2) add a standalone `probe_*.py`,
(3) archive the run JSON in `runs/` and the verdict in
`results/RESULTS_*.md`, (4) only then edit the paper. Every number in
either paper resolves to a JSON in `runs/`; do not edit paper numbers
without a matching run record.

## Run records
`runs/` — the per-run JSON archive (one file per pre-registered run,
including the early-stage records not indexed in `results/RESULTS_*.md`:
the full-split COCO-Object/Context-60 transfer baselines
`fullpix_*`/`full_*`, the oracle AUC and SLIC/veto probes, and the failed
guard variants); every number in either paper resolves to exactly one
JSON in this directory.

## Scope notes
Most probes are single-seed on frozen test-300 / heldout-200 subsets;
the papers state the applicable subset for every number. Nothing in
this artifact was tuned after unblinding a result; where a criterion
was later found ill-posed (e.g. the presence-gating mIoU bar), the
verdict was kept and the issue disclosed rather than re-judged.
