#!/bin/bash
# W5 queue: W5a presence2 (6 SAM runs) then W5e ba_conf (42 confusion runs)
set -u
cd /media/dell/DATA/ovss/research_run/ovss/stage4_experiments/vocabaudit
PY=/media/dell/DATA/ovss/miniconda3/envs/ovss/bin/python
RUNS=/media/dell/DATA/ovss/runs
SAM=/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth

# ---- W5a presence-gated REVA v2
for m in clearclip naclip; do
  for v in official plain dis_near200; do
    out=$RUNS/w5a_${m}_${v}.json
    [ -f "$out" ] && continue
    $PY probe_presence2.py --variant $m --dataset voc21 --limit 300 \
      --vocab-file perturbed_vocabs/voc21_${v}.json \
      --neg-file perturbed_vocabs/voc21_plain_vabs64_meta.json \
      --n-gt-classes 21 --sam-ckpt $SAM --out $out \
      >> $RUNS/w5a.log 2>&1
  done
done

# ---- W5e confusion capture
for m in maskclip sclip clearclip naclip proxyclip lposs scclip; do
  for ds in voc21 coco171; do
    ng=21; [ $ds = coco171 ] && ng=171
    for v in plain syn100_s0 dis_near200; do
      out=$RUNS/w5e_${m}_${ds}_${v}.npz
      [ -f "$out" ] && continue
      $PY probe_ba_conf.py --variant $m --dataset $ds --limit 300 \
        --vocab-file perturbed_vocabs/${ds}_${v}.json \
        --n-gt-classes $ng --out $out >> $RUNS/w5e.log 2>&1
    done
  done
done
echo "W5 queue done $(date)"
