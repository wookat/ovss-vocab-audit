#!/bin/bash
# VOC-21 audit queue (waits for base_* queue to finish via GPU free check)
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
run() {  # name variant dataset extra...
  local name=$1; shift
  [ -f "$R/$name.json" ] && { echo "skip $name"; return; }
  python run_eval.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1
  echo "done $name: $(grep -o '"miou": [0-9.]*' $R/$name.json)"
}
V=voc21; L=300
for variant in sclip clearclip maskclip; do
  run ${V}_${variant}_plain        --variant $variant --dataset $V --limit $L --vocab-file perturbed_vocabs/${V}_plain.json
  run ${V}_${variant}_plain_zca    --variant $variant --dataset $V --limit $L --vocab-file perturbed_vocabs/${V}_plain.json --whiten zca
  run ${V}_${variant}_plain_center --variant $variant --dataset $V --limit $L --vocab-file perturbed_vocabs/${V}_plain.json --whiten center
  for p in syn25_s0 syn50_s0 syn100_s0 syn50_s1 syn50_s2 dis_near50 dis_near200 dis_mid50; do
    run ${V}_${variant}_${p}     --variant $variant --dataset $V --limit $L --vocab-file perturbed_vocabs/${V}_${p}.json
    run ${V}_${variant}_${p}_zca --variant $variant --dataset $V --limit $L --vocab-file perturbed_vocabs/${V}_${p}.json --whiten zca
  done
done
echo ALL-DONE
