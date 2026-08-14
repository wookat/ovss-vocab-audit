# P3-E3: detection-specific coupling go/no-go (frozen before runs)

Precondition C2 of the paper-3 evaluation card: does the detection stack
couple with vocabulary perturbation in ways segmentation cannot exhibit
(NMS/decoding/score-threshold effects)? This is what separates the third
paper from a re-run of the segmentation audit.

## Protocol (OWLv2-base, COCO val2017 first 1000 ids, as E1)
E3a synonym-coexistence: vocabulary = plain 80 + syn1st 80 (160 queries;
query i and i+80 both map to COCO category i). Measured:
 (1) mAP of the coexistence vocabulary with the model's default per-class
     handling vs plain-80 mAP;
 (2) duplicate rate: fraction of kept detections having IoU>=0.9 with a
     detection of the same category originating from the other name variant;
 (3) mAP after cross-name class-wise NMS merge (IoU 0.65) of the two name
     variants' detections.
E3b drop decomposition: from raw detections of E1 plain and syn1st arms
(re-run with detections dumped, threshold 0.001), decompose the synonym AP
drop at IoU=0.5: recall-loss share = GT boxes matched (IoU>=0.5, any score)
under plain but unmatched under syn1st; the remainder of the AP drop is
attributed to score degradation/reordering of still-detected objects.

## Frozen criteria (from the evaluation card)
- GO: coupling effect >= 3 AP (|coexist - plain| or |merged - coexist|)
  OR duplicate rate >= 15%.
- NO-GO: neither threshold met -> detection-specific phenomenon space too
  thin; third paper degrades to an "audit v2 extension section".
- E3b is descriptive (no criterion); it feeds the mechanism section either
  way.
Single run; subset and thresholds as E1; disclosures as usual.
