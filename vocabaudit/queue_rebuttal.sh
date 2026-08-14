#!/bin/bash
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
run() { local name=$1; shift; [ -f "$R/$name.json" ] && { echo "skip $name"; return; }; python run_eval.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1; echo "done $name: $(grep -o '"miou[_al]*": [0-9.]*' $R/$name.json | tr '\n' ' ')"; }
# R2-Q4: distractor effect with engineered background baseline
for v in sclip clearclip maskclip; do
  run reb_voc21_${v}_official           --variant $v --dataset voc21 --limit 300
  run reb_voc21_${v}_official_dis200    --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_official_dis_near200.json
done
# full-split calibration (VOC val 1449 imgs)
for v in sclip clearclip maskclip; do
  run reb_voc21_${v}_plain_full    --variant $v --dataset voc21 --vocab-file perturbed_vocabs/voc21_plain.json
  run reb_voc21_${v}_official_full --variant $v --dataset voc21
done
# subset resampling variance (plain, 4 disjoint 300-img subsets)
for v in sclip clearclip; do
  for off in 0 300 600 900; do
    run reb_voc21_${v}_plain_off$off --variant $v --dataset voc21 --limit 300 --offset $off --vocab-file perturbed_vocabs/voc21_plain.json
  done
done
# distractor runs re-emitted with miou_all (standard all-class mIoU)
for v in sclip clearclip maskclip; do
  run reb_voc21_${v}_dis_near200_v2 --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_dis_near200.json
done
echo REBUTTAL-DONE
