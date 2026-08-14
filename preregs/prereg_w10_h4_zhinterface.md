# Pre-registration W10-H4: Chinese collapse — interface artifact vs capability (frozen before run)

Date frozen: 2026-08-01, after W7c/W8 (zh floor collapse observed) and
before any adaptation run.

## Question
Is the Chinese floor collapse (<10 mIoU on both SCLIP and OWLv2) a real
multilingual-capability deficit, or partly an interface artifact (token
overflow / prompt-template mismatch), in the spirit of the BA decomposition
(rule out artifacts before interpreting)?

## Adaptations (frozen; no tuning after results)
On VOC test-300, zh vocabulary from W7c, models SCLIP and OWLv2+SAM:
- A1 bare name: query the Chinese name with NO English template
  ("{name}" instead of "a photo of a {name}."), removing template/token
  pressure.
- A2 romanization-free short form: first two characters of each name
  (subword-pressure reduction), same template as A1.
- A3 back-translation: the frozen English plain name (upper reference =
  the plain row; this arm just re-anchors the ceiling; no new run needed).

## Criteria (frozen)
- ARTIFACT-DOMINATED: any adaptation recovers >= 15 mIoU over the W7c zh
  cell on either model.
- CAPABILITY-DEFICIT: all adaptations gain < 5 -> the collapse is real
  multilingual incapacity of the English-BPE text towers; keep the
  boundary-note wording.
- MIXED: gains in [5, 15).

## Cost
4 runs, < 0.5 GPU day.
