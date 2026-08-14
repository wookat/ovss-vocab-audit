# Pre-registration W7b: text-encoder-lineage go/no-go (frozen before run)

Date frozen: 2026-08-01. Round-7 candidate G1. Question: does the F2
synonym-fragility contrast (BERT-grounded GDINO -37.6 vs CLIP-tower OWLv2
-7.1 on VOC) generalize to a third model per lineage, i.e. is fragility
banded by text-encoder lineage rather than idiosyncratic to the two models?

## Setup (frozen)
Third models, same box->SAM-B harness as F2 (per-class/chunked queries,
score arbitration, unclaimed = background):
- BERT/RoBERTa-grounded: GLIP-T (if transformers/weights unavailable
  offline within a day, fallback MDETR; the substitution must be recorded
  before results are seen).
- CLIP-text-tower: OWL-ViT v1 (google/owlvit-base-patch32).
Dataset: VOC-21 test-300. Vocabularies: plain, syn100_s0. Metric:
GT-present mIoU delta (plain - synonym), within-detector.

## Criteria (frozen)
Band definitions from F2: BERT-grounded band = synonym drop >= 20;
CLIP-tower band = synonym drop <= 15.
- GO (lineage holds): both third models fall in their lineage band ->
  expand to full 6-model matrix + token-level mechanism probes (subword
  perturbation sensitivity, layerwise synonym drift) + translation axis.
- NO-GO (lineage broken): either model lands outside its band -> the
  lineage hypothesis is falsified as stated; report as an honest negative
  and re-frame ("fragility determined by something else"); no silent
  re-banding.
- KILL (infeasible): a third model cannot reach plain GT-present >= 35 on
  VOC in this harness -> that model is excluded (recorded), try fallback;
  if neither lineage can field a third model, record infeasibility.

## Cost
Pure inference, ~0.5-1 GPU day.
