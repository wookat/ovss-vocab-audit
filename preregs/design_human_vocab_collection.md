# Design: human-sourced plain-vocabulary collection (ready-to-run kit)

Status: DESIGN ONLY (2026-08-04). This is the single remaining item in
both papers that cannot be closed internally — it needs real human
participants. This document freezes the protocol so collection can
start the day the boss approves it; no analysis choices are left to be
made after data arrives.

## 1. Claim at stake
Both papers currently construct "plain" vocabularies by a frozen
mechanical rule (first WordNet lemma / dataset class name stripped of
engineering). Reviewers can ask: do real lay users actually type names
like these? The audit's headline deltas (official vs plain) would be
strengthened (or corrected) by vocabularies elicited from humans.

## 2. Elicitation protocol (frozen)
- Task: participant sees the DATASET CLASS DEFINITION (one-sentence
  gloss + 3 example crops per class, crops sampled by frozen seed from
  train split — never val), and types "the word or short phrase you
  would use to ask a segmentation tool to find this".
- No priming with any existing class name; glosses are paraphrased to
  avoid quoting the official name (gloss file frozen and archived
  before collection).
- Datasets: VOC-21 (20 classes) and COCO-Object (80 classes).
- N = 15 participants minimum per dataset (power: with 15, a per-class
  majority name is stable under leave-one-out for >= 90% of classes in
  pilot simulations; see 5).
- Recruitment: colleagues/students NOT in the project, or a crowd
  platform; record only coarse demographics (native language y/n,
  CV background y/n). No PII. Compensation per local norms.

## 3. Vocabulary construction (frozen)
- V-majority: per class, the modal string (lowercased, whitespace
  normalized). Ties broken by earliest submission.
- V-individual_i: each participant's full vocabulary — evaluated
  separately to report BETWEEN-USER variance of mIoU (the number no
  mechanical rule can produce).
- Refusals/empty answers keep the mechanical plain name (logged).

## 4. Evaluation (frozen)
- Hosts: SCLIP, NACLIP (local reimpl, full protocol) + Trident official
  code (anchor). Full val sets, single seed, same harness as the paper.
- Report: official vs mechanical-plain vs V-majority vs the 15
  V-individual curves (min/median/max).
- Pre-declared hypotheses:
  - H1: |mIoU(V-majority) - mIoU(mechanical plain)| <= 3 macro points
    on each host -> the mechanical rule is a fair proxy (papers keep
    current numbers, add a validation sentence).
  - H2: between-user spread (max-min over V-individual) >= 5 points on
    any host -> naming variance among real users is itself material;
    reported as a new finding either way (descriptive).
  - If H1 fails, the audit's plain rows are RELABELED as "mechanical
    plain" and V-majority becomes the headline plain row; deltas
    recomputed; no silent substitution.

## 5. What can be done before humans arrive (internal, optional)
- Freeze gloss files + crop seeds; dry-run the form end-to-end with 2
  project-internal fillers (their data marked PILOT, excluded from
  analysis by rule).
- Simulation for the N=15 stability claim: bootstrap over WordNet
  lemma distributions — internal calibration only, never a substitute
  for the human data and never citable as such.

## 6. Explicit non-substitution rule
LLM-generated "user" vocabularies are NOT an acceptable stand-in and
will not be run as a fallback: the entire point is out-of-model naming
behavior. If collection is not approved, the papers keep the current
honest caveat.

## 7. Cost estimate
15 x 2 datasets x ~15 min ≈ 8 person-hours of participant time; eval
compute ~1 GPU-day (full val, 3 hosts x ~17 vocabularies). All harness
code already exists.
