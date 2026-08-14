#!/bin/bash
# W6 queue: F3 law (offline), F1 RECAL grid, F2 cross-family audit
set -u
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
PY=/media/dell/DATA/ovss/miniconda3/envs/ovss/bin/python
RUNS=/media/dell/DATA/ovss/runs
SAM=/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth
CACHE=/media/dell/DATA/ovss/recal_cache

# ---- F3 law (text-only, minutes)
[ -f $RUNS/w6f3_law.json ] || \
  $PY probe_law.py --out $RUNS/w6f3_law.json >> $RUNS/w6.log 2>&1

# ---- F1 RECAL: 2 methods x {plain + 3 synonym vocabs}
for m in sclip naclip; do
  for v in plain syn100_s0 syn100_s1 syn100_s2; do
    out=$RUNS/w6f1_${m}_${v}.json
    [ -f "$out" ] && continue
    $PY probe_recal.py --variant $m --dataset voc21 --limit 300 \
      --vocab-file perturbed_vocabs/voc21_${v}.json \
      --cache-dir $CACHE/${m}_${v} --out $out >> $RUNS/w6.log 2>&1
    rm -rf $CACHE/${m}_${v}
  done
done

# ---- F2 cross-family: GDINO+SAM x {plain, syn100_s0, dis_near200}
for v in plain syn100_s0 dis_near200; do
  out=$RUNS/w6f2_gdino_${v}.json
  [ -f "$out" ] && continue
  $PY probe_crossfam.py --dataset voc21 --limit 300 \
    --vocab-file perturbed_vocabs/voc21_${v}.json \
    --n-gt-classes 21 --sam-ckpt $SAM --out $out >> $RUNS/w6.log 2>&1
done

echo "W6 queue done $(date)" >> $RUNS/w6.log
