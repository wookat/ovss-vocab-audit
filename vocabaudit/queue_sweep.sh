#!/bin/bash
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
run() { local name=$1; shift; [ -f "$R/$name.json" ] && { echo "skip $name"; return; }; python run_eval.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1; echo "done $name: $(grep -o '"miou": [0-9.]*' $R/$name.json)"; }
for s in 0.1 0.3 0.7 0.9; do
  run sweep_pc459_clearclip_zca_s$s --variant clearclip --dataset pc459 --limit 500 --whiten zca --shrink $s
done
for v in sclip clearclip maskclip; do
  run kc_ade150_${v}_none   --variant $v --dataset ade150 --limit 500
  run kc_ade150_${v}_center --variant $v --dataset ade150 --limit 500 --whiten center
  run kc_ade150_${v}_zca    --variant $v --dataset ade150 --limit 500 --whiten zca
  run kc_coco171_${v}_none   --variant $v --dataset coco171 --limit 500
  run kc_coco171_${v}_center --variant $v --dataset coco171 --limit 500 --whiten center
  run kc_coco171_${v}_zca    --variant $v --dataset coco171 --limit 500 --whiten zca
  run voc21_${v}_gran_coarse  --variant $v --dataset voc21  --limit 300 --vocab-file perturbed_vocabs/voc21_gran_coarse.json
  run ade150_${v}_gran_coarse --variant $v --dataset ade150 --limit 500 --vocab-file perturbed_vocabs/ade150_gran_coarse.json
done
echo SWEEP-DONE
