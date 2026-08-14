#!/bin/bash
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
run() { local name=$1; shift; [ -f "$R/$name.json" ] && { echo "skip $name"; return; }; python run_eval.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1; echo "done $name: $(grep -o '"miou": [0-9.]*' $R/$name.json)"; }
for v in sclip clearclip maskclip; do
  run reb2_ade150_${v}_globalzca  --variant $v --dataset ade150  --limit 300 --whiten zca --stats-vocab perturbed_vocabs/globalstats_ade847_coco171.json
  run reb2_coco171_${v}_globalzca --variant $v --dataset coco171 --limit 300 --whiten zca --stats-vocab perturbed_vocabs/globalstats_ade847_coco171.json
done
echo REBUTTAL2-DONE
