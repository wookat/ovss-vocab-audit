# Pre-registration W10 addendum: Spanish best-of-k translation control (frozen before run)

Date frozen: 2026-08-01, in response to simulated-R2's W8 objection: the
52-62% dense retention band rests on one dictionary translation per class,
confounding "translation choice quality" with "multilingual capability".

## Design (frozen)
For each VOC-21 class, 3 legal Spanish translations (the frozen W7c/W8
primary plus 2 dictionary-sanctioned alternates, fixed below before any
run). Models: SCLIP (dense representative) and OWLv2+SAM. VOC test-300.
Report per-class best-of-3 (oracle over translations, upper bound) mIoU
and its retention vs plain English, next to the single-translation cell.

Alternates (frozen):
avion: aeroplano, avioneta          bicicleta: bici, velocipedo
pajaro: ave, pajarito               barco: bote, embarcacion
botella: frasco, envase             autobus: bus, camion de pasajeros
coche: carro, automovil             gato: minino, felino
silla: asiento, butaca              vaca: res, bovino
mesa de comedor: mesa, comedor      perro: can, cachorro
caballo: equino, corcel             motocicleta: moto, motociclo
persona: humano, gente              planta en maceta: planta, maceta
oveja: cordero, borrego             sofa: sillon, canape
tren: ferrocarril, locomotora       televisor: television, tele
(fondo/background unchanged)

## Criteria (frozen)
- CAPABILITY: best-of-3 dense retention stays < 70% -> the 52-62% band is
  capability-limited, not translation-choice-limited; quantitative claim
  stands with the control disclosed.
- TRANSLATION-CHOICE: best-of-3 retention >= 85% -> rewrite the section:
  single-translation fragility, not multilingual incapacity.
- MIXED: in between -> report both mechanisms, soften the band claim.

Note best-of-3 is an oracle upper bound (per-class max), so this test is
biased AGAINST the capability reading; surviving it is strong evidence.
