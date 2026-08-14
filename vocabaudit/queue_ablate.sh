#!/bin/bash
# waits for full queue then runs scoop-check ablations on VOC test-300
set -x
cd "$(dirname "$0")"
R=/media/dell/DATA/ovss/runs
CK=/media/dell/DATA/ovss/checkpoints/sam_vit_b_01ec64.pth
PV=perturbed_vocabs
while ! grep -q FULL_DONE $R/full_queue.log; do sleep 120; done
python - <<'PYEOF'
import json, data
plain = json.load(open("perturbed_vocabs/voc21_plain.json"))
official = data.voc21()[1]
hand = list(plain); hand[0] = official[0]
json.dump(hand, open("perturbed_vocabs/voc21_plain_handbg.json", "w"))
print("handbg bg entry:", hand[0][:100])
PYEOF
for M in sclip clearclip maskclip naclip; do
  python probe_ablate.py --variant $M --dataset voc21 --offset 0 --limit 300 \
    --vabs-vocab $PV/voc21_plain_vabs64.json --handbg-vocab $PV/voc21_plain_handbg.json \
    --sam-ckpt $CK --out $R/abl_voc_${M}_test300.json
done
echo ABL_DONE
