#!/bin/bash
# review-fix queue: dev-excluded (1349-image) main table + full-split ablation controls
set -x
cd "$(dirname "$0")"
R=/media/dell/DATA/ovss/runs
CK=/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth
PV=perturbed_vocabs
for M in sclip clearclip maskclip naclip; do
  python probe_d1sam.py --variant $M --dataset voc21 --offset 0 --limit 100000 --skip-dev \
    --vabs-vocab $PV/voc21_plain_vabs64.json --rand-vocab $PV/voc21_plain_randneg64.json \
    --sam-ckpt $CK --out $R/clean_voc_${M}.json
done
for M in sclip naclip; do
  python probe_d1sam.py --variant $M --dataset voc21 --offset 0 --limit 100000 --skip-dev \
    --vabs-vocab $PV/voc21_official.json --rand-vocab $PV/voc21_official.json \
    --sam-ckpt $CK --out $R/clean_voc_${M}_official.json
done
for M in sclip clearclip maskclip naclip; do
  python probe_ablate.py --variant $M --dataset voc21 --offset 0 --limit 100000 --skip-dev \
    --vabs-vocab $PV/voc21_plain_vabs64.json --handbg-vocab $PV/voc21_plain_handbg.json \
    --sam-ckpt $CK --out $R/clean_abl_voc_${M}.json
done
echo CLEAN_DONE
