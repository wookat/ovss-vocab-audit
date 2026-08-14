#!/bin/bash
# prereg_v5 full validation queue (sequential, single 3090)
set -x
cd "$(dirname "$0")"
R=/media/dell/DATA/ovss/runs
CK=/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth
PV=perturbed_vocabs

# vocab prep
python - <<'EOF'
import json, data, os
os.makedirs("perturbed_vocabs", exist_ok=True)
json.dump(data.voc21()[1], open("perturbed_vocabs/voc21_official.json", "w"))
json.dump(data.ctx60()[1], open("perturbed_vocabs/ctx60_plain.json", "w"))
EOF
[ -f $PV/ctx60_plain_vabs64.json ] || python vabs.py --vocab-file $PV/ctx60_plain.json --lexicon scene --M 64 --tau-sim 0.90 --out $PV/ctx60_plain_vabs64.json
[ -f $PV/ctx60_plain_randneg64.json ] || python vabs.py --vocab-file $PV/ctx60_plain.json --lexicon scene --M 64 --tau-sim 0.90 --random --seed 0 --out $PV/ctx60_plain_randneg64.json

# 1) VOC test-300 x 4 methods
for M in sclip clearclip maskclip naclip; do
  python probe_d1sam.py --variant $M --dataset voc21 --offset 0 --limit 300 \
    --vabs-vocab $PV/voc21_plain_vabs64.json --rand-vocab $PV/voc21_plain_randneg64.json \
    --sam-ckpt $CK --out $R/v5_voc_${M}_test300.json
done

# 2) V5 official safety check (SCLIP+NACLIP): official vocab in both slots
for M in sclip naclip; do
  python probe_d1sam.py --variant $M --dataset voc21 --offset 0 --limit 300 \
    --vabs-vocab $PV/voc21_official.json --rand-vocab $PV/voc21_official.json \
    --sam-ckpt $CK --out $R/v5_voc_${M}_official_test300.json
done

# 3) transfer: COCO-Object + Context-60 (SCLIP+NACLIP)
for M in sclip naclip; do
  python probe_d1sam.py --variant $M --dataset cocoobj --offset 0 --limit 300 \
    --vabs-vocab $PV/cocoobj_plain_vabs.json --rand-vocab $PV/cocoobj_plain_randneg64.json \
    --sam-ckpt $CK --out $R/v5_cocoobj_${M}_test300.json
  python probe_d1sam.py --variant $M --dataset ctx60 --offset 0 --limit 300 \
    --vabs-vocab $PV/ctx60_plain_vabs64.json --rand-vocab $PV/ctx60_plain_randneg64.json \
    --sam-ckpt $CK --out $R/v5_ctx60_${M}_test300.json
done
echo QUEUE_DONE
