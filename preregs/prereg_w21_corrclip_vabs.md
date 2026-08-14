# Prereg W21: VABS component validation on official CorrCLIP (frozen before runs)

Date frozen: 2026-08-03 (before any W21 arm is launched)

## Motivation
CorrCLIP (ICCV 2025 Oral) is the sixth audit anchor and the training-free
performance-ceiling representative (SAM2 region masks + MetaCLIP + DINO).
W20 measured NEG +20.5 (VOC-21) and +1.4 (COCO-Object) on the official
code. This preregisters the REVA/VABS component test on the same official
stack, extending the gap-vs-recovery dose curve to a fourth official
codebase and a second SAM-coupled host.

## Protocol (frozen)
- Official CorrCLIP repo, forward unmodified; only the class-name file is
  changed. VABS-64 negatives are appended to the background line of the
  plain class file using the repo's own '; ' alias mechanism (identical to
  how the official file packs 16-26 background sub-classes).
- VABS negatives: the frozen W16-C recipe conditioned on the plain
  vocabulary of each dataset (same files as used for SC-CLIP/Trident
  official validations where dataset matches).
- Matched random-negative control: 64 random words, seed 0, same file
  mechanics (same files as prior official validations where applicable).
- Datasets: VOC-21 and COCO-Object (both passed the W20 official
  reproduction gate). Context-60 excluded (failed gate in W20).
- Single run per arm. Region masks identical across arms by construction.

## Arms (4 new; plain baselines reuse W20 logs)
1. voc21 plain+VABS64
2. voc21 plain+rand64 (seed 0)
3. cocoobj plain+VABS64
4. cocoobj plain+rand64 (seed 0)

## Frozen predictions and criteria (from the gap-vs-recovery boundary model)
- VOC-21 (NEG +20.5, large headroom): VABS gain over plain >= +3.0 mIoU
  AND selection advantage (VABS - rand) > 0. Expected qualitative match to
  SC-CLIP (+16.7) and Trident (+16.0) magnitudes is NOT a criterion, only
  direction and the two thresholds.
- COCO-Object (NEG +1.4, near-zero headroom): prediction is parity or
  harm: |VABS - rand| interpreted descriptively; VABS - plain expected
  < +3.0. If VABS - plain >= +3.0 here, the boundary model takes a hit and
  this must be reported as evidence against it.
- Claim scope: VABS-only component evidence; no full-REVA (SAM arbitration)
  claim on CorrCLIP.

## Outcome handling
Results reported regardless of direction; failures preserved as failures.
