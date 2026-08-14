# Pre-registration W12-K1a: absent-class leakage mechanism of synonym damage (frozen before any analysis)

Date frozen: 2026-08-01, after J5 GO and before any confusion-flow
analysis of the synonym axis with respect to class presence.

## Hypothesis (from tenth-round ideator K1)
Synonym renaming damages dense CLIP mainly by leaking pixels to classes
ABSENT from the image: renaming weakens the true class's margin, and the
first winners are absent classes with no pixel support competing purely
on text similarity. If true, J5 pruning (which removes absent classes) is
a mechanism-aligned repair, not a trick — and its synonym-arm gain is
explained.

## Frozen analysis
Data: existing/new per-image confusion flows on VOC test-300, SCLIP,
plain vs syn100_s0 (re-run saving per-image assignments if archived
conf matrices are pooled — pooled matrices cannot separate present vs
absent per image, so a re-run with per-image accounting is allowed; the
evaluation protocol is unchanged).
Measure, per image: of the GT-foreground pixels that CHANGE away from
their correct class when going plain -> syn100, the fraction assigned to
(a) classes present in the image's GT, (b) classes absent from it,
(c) background.
Class-level: correlation (Spearman) between per-class J5 pruning gain on
the synonym arm and the per-class absent-leak fraction.

## Criteria (frozen)
- GO (mechanism leg): pooled absent-class share of leaked foreground
  pixels >= 50% AND class-level Spearman >= 0.4.
- NO-GO: absent share < 35% or Spearman < 0.25.
- MIXED: between.

## Scope
One model (SCLIP), one dataset (VOC), one synonym seed; if GO, replicate
on ClearCLIP before paper claims.
