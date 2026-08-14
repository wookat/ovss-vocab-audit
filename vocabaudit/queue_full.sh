#!/bin/bash
# step-3 full-split validation (v5-final form)
set -x
cd "$(dirname "$0")"
R=/media/dell/DATA/ovss/runs
CK=/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth
PV=perturbed_vocabs
for M in sclip clearclip maskclip naclip; do
  python probe_d1sam.py --variant $M --dataset voc21 --offset 0 --limit 100000 \
    --vabs-vocab $PV/voc21_plain_vabs64.json --rand-vocab $PV/voc21_plain_randneg64.json \
    --sam-ckpt $CK --out $R/full_voc_${M}.json
done
for M in sclip naclip; do
  python probe_d1sam.py --variant $M --dataset voc21 --offset 0 --limit 100000 \
    --vabs-vocab $PV/voc21_official.json --rand-vocab $PV/voc21_official.json \
    --sam-ckpt $CK --out $R/full_voc_${M}_official.json
  python probe_d1sam.py --variant $M --dataset cocoobj --offset 0 --limit 100000 \
    --vabs-vocab $PV/cocoobj_plain_vabs.json --rand-vocab $PV/cocoobj_plain_randneg64.json \
    --sam-ckpt $CK --out $R/full_cocoobj_${M}.json
  python probe_d1sam.py --variant $M --dataset ctx60 --offset 0 --limit 100000 \
    --vabs-vocab $PV/ctx60_plain_vabs64.json --rand-vocab $PV/ctx60_plain_randneg64.json \
    --sam-ckpt $CK --out $R/full_ctx60_${M}.json
done
echo FULL_DONE
