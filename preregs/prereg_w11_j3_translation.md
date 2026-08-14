# Pre-registration W11-J3: translation-choice sensitivity beyond Spanish (frozen before any run)

Date frozen: 2026-08-01, after W10's Spanish best-of-3 verdict
(TRANSLATION-CHOICE) and before any German/Russian run.

## Question
Is the oracle-vs-default translation gap a Spanish quirk or a general
property of multilingual OVSS evaluation? If general, multilingual
reports should carry a [default, best-of-k, worst-of-k] triple.

## Design (frozen)
Languages: German, Russian (Spanish already done in W10). k=3 dictionary
translations per VOC-20 class, frozen below. Models: SCLIP (VOC
test-300, dense protocol) and OWLv2+SAM (VOC test-300 box->SAM
protocol). Metric: GT-present mIoU; retention vs plain English; per-class
best-of-3 oracle; oracle-default gap in retention percentage points.

German (primary / alt1 / alt2):
Flugzeug/Flieger/Maschine; Fahrrad/Rad/Velo; Vogel/Piepmatz/Federvieh;
Boot/Schiff/Kahn; Flasche/Buddel/Pulle; Bus/Omnibus/Reisebus;
Auto/Wagen/PKW; Katze/Kater/Mieze; Stuhl/Sessel/Sitz; Kuh/Rind/Milchkuh;
Esstisch/Tisch/Tafel; Hund/Koeter/Welpe; Pferd/Gaul/Ross;
Motorrad/Maschine/Kraftrad; Person/Mensch/Leute;
Topfpflanze/Zimmerpflanze/Pflanze; Schaf/Lamm/Schafbock;
Sofa/Couch/Kanapee; Zug/Eisenbahn/Bahn; Fernseher/Fernsehgeraet/TV-Geraet
(umlauts written properly in the vocab files).

Russian (primary / alt1 / alt2):
samolyot/aeroplan/layner; velosiped/velik/baik; ptitsa/ptichka/pernatoye;
lodka/sudno/korabl; butylka/flakon/butyl; avtobus/marshrutka/mikroavtobus;
mashina/avtomobil/avto; koshka/kot/kotyonok; stul/kreslo/sideniye;
korova/byk/burenka; obedenny stol/stol/stolik; sobaka/pyos/shchenok;
loshad/kon/zherebets; mototsikl/motobaik/baik; chelovek/persona/lyudi;
komnatnoye rasteniye/rasteniye v gorshke/tsvetok v gorshke;
ovtsa/baran/yagnyonok; divan/sofa/kushetka; poezd/elektrichka/sostav;
televizor/TV/teleekran (Cyrillic in the vocab files).

## Criteria (frozen, per ideator draft)
- GO: >= 2 of the 3 languages (counting Spanish) show an oracle-default
  retention gap >= 15 percentage points on at least one model family ->
  translation-choice sensitivity is a general protocol hazard; promote to
  a protocol recommendation in the paper.
- NO-GO: new-language gaps < 8 points -> Spanish is an outlier; fold into
  the existing W10 note, no protocol claim.
- MIXED: in between.
- Degradation kill: if a language's gap is driven by a single mistranslated
  outlier (removing the worst alternate leaves gap < 5), treat as data
  cleaning, not a protocol story.

## Scope guards
Dictionary translations frozen before runs; Latin transliteration above is
for this file only, actual vocabs use native script; one language per run;
no capability claims (H4's Chinese remains the capability anchor).
