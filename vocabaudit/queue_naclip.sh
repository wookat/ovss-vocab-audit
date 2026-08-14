#!/bin/bash
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
run() { local name=$1; shift; [ -f "$R/$name.json" ] && { echo "skip $name"; return; }; python run_eval.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1; echo "done $name: $(grep -o '"miou": [0-9.]*' $R/$name.json)"; }
v=naclip
run nac_voc21_official       --variant $v --dataset voc21 --limit 300
run nac_voc21_plain          --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_plain.json
run nac_voc21_syn50_s0       --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_syn50_s0.json
run nac_voc21_syn50_s1       --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_syn50_s1.json
run nac_voc21_syn50_s2       --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_syn50_s2.json
run nac_voc21_syn100         --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_syn100_s0.json
run nac_voc21_gran           --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_gran_coarse.json
run nac_voc21_dis_near200    --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_dis_near200.json
run nac_voc21_official_dis200 --variant $v --dataset voc21 --limit 300 --vocab-file perturbed_vocabs/voc21_official_dis_near200.json
run nac_pc459_plain          --variant $v --dataset pc459 --limit 500
run nac_pc459_zca            --variant $v --dataset pc459 --limit 500 --whiten zca
run nac_pc459_center         --variant $v --dataset pc459 --limit 500 --whiten center
run nac_ade150_plain         --variant $v --dataset ade150 --limit 300 --vocab-file perturbed_vocabs/ade150_plain.json
run nac_ade150_gran          --variant $v --dataset ade150 --limit 300 --vocab-file perturbed_vocabs/ade150_gran_coarse.json
echo NACLIP-DONE
