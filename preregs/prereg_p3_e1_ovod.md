# P3-E1: OVOD vocabulary-audit go/no-go — phenomenon existence (frozen before runs)

Third-paper conditional-GO precondition C1 (see stage6_review's paper-3
evaluation card): does vocabulary/naming perturbation move NATIVE detection
AP (not our seg conversion) on open-vocabulary detectors?

## Protocol
- Models: OWLv2 (google/owlv2-base-patch16-ensemble) and YOLO-World
  (yolov8l-worldv2 via ultralytics), pure inference, single 3090.
- Data: COCO val2017, official instances_val2017.json, pycocotools
  COCO-style mAP (AP@[.5:.95]) over the 80 thing classes.
- Subset: first 1000 images of val2017 sorted by id (go/no-go scale;
  disclosed; full val only if the direction is pursued).
- Arms (same 80-class index order, only names change):
  1. plain: canonical COCO names;
  2. syn100 s0: the frozen WordNet+cosine synonym replacement already used
     for W16-D/W17-B (cls_coco_object_syn100.txt rows 1..80);
  3. syn100-2nd: at 100% replacement the frozen generator is deterministic
     (best valid synonym per class), so a "second seed" does not exist by
     design; instead the second synonym vocabulary takes the SECOND-ranked
     valid synonym per class (same WordNet + cosine [0.70,0.95] rule;
     classes with only one valid synonym keep the first), generated before
     any evaluation;
  4. distractor: plain + 200 frozen near-layer distractor words appended as
     extra queries (mapped to no COCO category; informational arm).
- Prompt template: each model's own default usage ("a photo of a {}" for
  OWLv2, bare names for YOLO-World set_classes). Score thresholds at each
  model's default; identical across arms.

## Frozen criteria (from the evaluation card, unchanged)
- GO: either model shows a synonym-axis relative mAP drop >= 15%
  (mean over the two synonym seeds vs plain).
- NO-GO: both models < 8% relative drop -> native AP is insensitive to
  naming; the third-paper line is killed and the negative result is noted
  in the audit paper as a one-liner.
- Between 8% and 15%: weak-GO, requires E3 (detection-specific coupling)
  to pass before any commitment.
- Distractor arm is reported descriptively; no criterion attaches to it.

Single run per arm; no seeds beyond the two synonym vocabularies; any
engineering workaround (weight transfer, chunking) disclosed.
