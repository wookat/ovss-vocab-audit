# SCI-CLIP anchor reproduction plan (frozen before any run, 2026-08-11)

Goal: satisfy the audit anchor protocol precondition for SCI-CLIP
(arXiv 2608.05627, official code github.com/mzamini92/SCICLIP, internal
name PRISM) before it can enter any formal comparison table. Reproduce
published values with the unmodified official repo (vocabulary/config
path edits only, per protocol).

## Frozen published targets (Table 2, CLIP ViT-B/16 block, mIoU)
Our audit's benchmark subset:
- VOC-21 (V21): 75.9
- COCO-Object (Object): 45.2
- Context-60 (PC60): 46.1
- ADE-150 (ADE): 29.8

Reproduction gate: same as prior official-code anchors — each cell must
match the published value within the protocol tolerance (|d| <= 0.5
mIoU) using official configs unmodified except data_root / weight-cache
paths. Any cell outside tolerance -> SCI-CLIP does NOT enter comparison
tables; record the deviation and environment diff instead.

## Pipeline (per official README)
1. conda env prism (py3.10), pip: eomt/requirements.txt + mmengine mmcv
   mmsegmentation openpyxl opencv-python hydra-core huggingface-hub.
2. SAM 2 Hiera-L checkpoint `sam2_hiera_large.pt` in repo root
   (download needed; GPU box is offline -> fetch on this box, transfer
   to /media/dell/DATA/ovss/weights/, symlink).
3. Per dataset: edit config data_root; build reference bank from TRAIN
   split first (bank builder swaps val.txt->train(aug).txt), point
   config at bank, then eval.py on val split.
4. Only after all four cells pass the gate: SCI-CLIP eligible for the
   audit vocabulary-perturbation protocol (plain/synonym/distractor)
   as a 7th official-code anchor.

## Dependencies / risks
- SAM2 Hiera-L (~2.3 GB) + CLIP/DINO weights must be staged offline.
- Reference-bank build over train splits is the heavy step (COCO/ADE
  train splits are large); budget >= 1 GPU-day total.
- Repo is a single 08-05 commit; watch for silent updates (tracked by
  intel session per 08-12 scan tasking).

Status (2026-08-11): setup phase started, no eval runs yet.
Status (2026-08-13): COMPLETE — verdict: gate FAILED (3/4 cells exact, ctx60 +1.56); SCI-CLIP excluded from formal tables. See RESULTS.md.
- sam2_hiera_large.pt downloaded (md5 08083462423be3260cd6a5eef94dc01c,
  897,952,466 bytes) -> /media/dell/DATA/ovss/weights/, symlinked into
  repo root.
- Official repo (single 08-05 commit) staged at
  /media/dell/DATA/ovss/code/SCICLIP.
- conda env `prism` (py3.10) created; pip install (torch 2.7.0 cu118 +
  eomt requirements + mm* stack) running, log
  /media/dell/DATA/ovss/logs/prism_pip.log.
- GPU box internet is back (pypi reachable); GPU currently shared with
  a PCOD process using ~14 GB of 24 GB -> paper used 4x L40s (48 GB);
  memory pressure is a real risk, may need to schedule around the
  other job or reduce batch. Record any deviation.
- base_config.py points dino_weights_path at a DINOv3 checkpoint not in
  the repo (dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth) -> must set
  to None (auto-download) or stage the weight; whichever is used must
  be recorded as part of the environment diff.
