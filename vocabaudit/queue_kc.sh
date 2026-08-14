#!/bin/bash
# KC1/KC2 decisive runs on PC-459 (large vocab). Runs BEFORE the VOC audit queue.
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
run() {
  local name=$1; shift
  [ -f "$R/$name.json" ] && { echo "skip $name"; return; }
  python run_eval.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1
  echo "done $name: $(grep -o '"miou": [0-9.]*' $R/$name.json)"
}
L=500
for variant in sclip clearclip maskclip; do
  run kc_pc459_${variant}_none   --variant $variant --dataset pc459 --limit $L
  run kc_pc459_${variant}_zca    --variant $variant --dataset pc459 --limit $L --whiten zca
  run kc_pc459_${variant}_center --variant $variant --dataset pc459 --limit $L --whiten center
  run kc_pc459_${variant}_global --variant $variant --dataset pc459 --limit $L --whiten zca --stats-vocab perturbed_vocabs/globalstats_ade847_coco171.json
done
echo KC-DONE
