#!/bin/bash
source /media/dell/DATA/ovss/miniconda3/etc/profile.d/conda.sh && conda activate ovss
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
R=/media/dell/DATA/ovss/runs
runc() { local name=$1; shift; [ -f "$R/$name.json" ] && { echo "skip $name"; return; }; python cls_control.py "$@" --out "$R/$name.json" > "$R/$name.log" 2>&1; echo "done $name: $(grep -o '"acc": [0-9.]*' $R/$name.json)"; }
runc cls_voc21_plain      --vocab-file perturbed_vocabs/voc21_plain.json
runc cls_voc21_syn50_s0   --vocab-file perturbed_vocabs/voc21_syn50_s0.json
runc cls_voc21_syn50_s1   --vocab-file perturbed_vocabs/voc21_syn50_s1.json
runc cls_voc21_syn50_s2   --vocab-file perturbed_vocabs/voc21_syn50_s2.json
runc cls_voc21_syn100_s0  --vocab-file perturbed_vocabs/voc21_syn100_s0.json
runc cls_voc21_dis_near200 --vocab-file perturbed_vocabs/voc21_dis_near200.json
runc cls_voc21_plain_zca  --vocab-file perturbed_vocabs/voc21_plain.json --whiten zca
echo CLS-DONE
