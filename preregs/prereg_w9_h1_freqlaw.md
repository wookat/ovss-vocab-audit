# Pre-registration W9 (H1): name-level frequency law go/no-go (frozen before analysis)

Date frozen: 2026-08-01, before any frequency estimate is attached to any
existing result.

## Question
Is per-name synonym damage predicted by the name's pretraining-corpus
frequency, within concept (concept fixed, only the name varies), across
models?

## Data (all already on disk; no new segmentation runs for the go/no-go)
Per-name damage observations: for each (model, class, name) cell where a
single class name was changed and per-class IoU deltas are recoverable
from archived confusion matrices / per-class tables:
- W7a freqctrl seeds (3 vocabs x ClearCLIP + OWLv2),
- ANS vocabularies (ClearCLIP, LPOSS search picks; heldout evals),
- syn100 suite per-class IoU tables where archived (per-class safety
  tables from W4h; robustbench per-class dumps where present).
Damage = per-class IoU(name) - IoU(plain name), same split.

## Frequency estimate (frozen)
Primary: word/phrase document frequency from a public English corpus
proxy accessible offline: Zipf frequency via the `wordfreq` package
(wordfreq.zipf_frequency(name, 'en'), multiword = min over tokens after
its own tokenization). If wordfreq is not installable offline, fallback:
log count in the local NLTK Brown+Reuters corpora (disclosed). No
switching between estimators after seeing correlations.

## Analysis (frozen)
Spearman correlation between damage and -zipf(name), within concept
pooled across classes, per model; report per-model rho and median.
Confounder check: partial Spearman controlling CLIP BPE token count.

## Criteria (frozen)
- GO: median per-model Spearman >= 0.5 across >= 5 models AND the
  within-concept relation holds (sign consistent in >= 70% of classes with
  >= 3 name observations) -> proceed to frequency-ladder vocabularies +
  stratified suite + H2 canonicalization prereg.
- NO-GO: median < 0.3 -> frequency is a proxy; do not build the law paper;
  record and pivot to H2 as a standalone repair test (its motivation
  weakens but the repair can still be tested against the observed damage).
- KILL (confound): partial rho controlling token count < 0.25 -> the
  "frequency" effect is tokenization; report as such.

## Boundary with dead directions
F3 killed text-embedding-geometry ranking of vocabularies; this uses an
external corpus statistic, not embedding geometry, and predicts per-name
damage within concept, not vocabulary-level ranking across methods.
