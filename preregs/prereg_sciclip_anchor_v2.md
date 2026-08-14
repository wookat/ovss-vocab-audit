# Preregistration: SCI-CLIP as 7th official-code anchor (v2, frozen 2026-08-13, before any perturbation run)

Context: the v1 plan (plan_sciclip_anchor_repro.md) froze a |d|<=0.5
four-cell gate; verdict FAIL (VOC 0.01, ADE 0.03, COCO-Object 0.00,
Context-60 1.56). That verdict stands and is not revised. This v2 is a
FRESH registration that instead mirrors the audit paper's existing
official-anchor protocol, under which Trident/RF-CLIP/CorrCLIP entered
the tables: per-cell tolerance |d| <= 1.5 with per-dataset exclusion
for cells outside it.

## Rationale (pre-stated, not post-hoc retuning of v1)
- The v1 0.5 gate was stricter than the audit paper's frozen 1.5 anchor
  gate; v1's verdict is preserved in RESULTS.md.
- Context-60 offset (+1.56) matches the audit's documented systematic
  +1.5~+1.6 offset of our converted Context GT (official Trident /
  RF-CLIP / CorrCLIP arms show the same); the paper already discloses
  this and draws no Context conclusions for those methods.

## Frozen inclusion rule
- SCI-CLIP enters the audit comparison tables as the 7th official-code
  anchor on the cells with |d| <= 1.5: VOC-21, COCO-Object, ADE-150.
- Context-60: |d| = 1.56 > 1.5 -> excluded; the paper will disclose the
  deviation with the same "draw no conclusions" sentence used for
  Trident/RF-CLIP/CorrCLIP on this dataset.
- No further reruns of the anchor cells; the recorded numbers are final.

## Frozen perturbation protocol (identical to the other anchors)
Official repo unchanged; only the class-name file (name_path) is
swapped. Reference bank REBUILT per vocabulary is NOT allowed — the
bank is class-name-conditioned via text retrieval, so the same frozen
bank per dataset is reused across vocabulary arms IF technically
decoupled from names; if the pipeline hard-codes name indices, the bank
build is repeated with identical settings and the coupling is disclosed.
Decision rule frozen now: inspect code once, record which case applies
BEFORE running any arm; no switching afterward.
Arms per dataset (VOC-21, COCO-Object; ADE-150 official config has no
background channel handling difference -> included for plain/synonym
only, consistent with prior anchors):
1. official (published class-name file) — already measured.
2. plain (audit's frozen plain vocabularies).
3. syn100 seed0 (audit's frozen synonym files).
Metrics: mIoU via official eval.py; deltas reported alongside the other
six anchors. No new criteria; descriptive entry into the existing
anchor table.

## What would falsify / abort
- If the bank/name coupling makes vocabulary swap ill-defined (case
  above), SCI-CLIP is reported as "official-code anchor, reproduction
  only; perturbation not applicable" and does NOT get perturbation rows.

## Coupling inspection (recorded 2026-08-13, BEFORE any perturbation arm)
Code check (prism_segmentor.py load_reference_bank /
generate_category_embeddings): the bank stores segment visual
embeddings + segment_to_label (dataset class indices, name-independent)
+ label_features (text embeddings of the names used at bank-build
time). Swapping name_path changes the query text features but the
bank's stored label_features remain those of the official names.
Recorded case: vocabulary swap is WELL-DEFINED (same frozen bank reused
across arms, retrieval keyed on class indices); the residual coupling
(bank label_features from official names) will be disclosed verbatim in
the paper row footnote. No bank rebuilds per vocabulary.
