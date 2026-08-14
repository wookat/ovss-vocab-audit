# Pre-registration W2a: ADE-150 REVA validation (frozen 2026-07-31)

Question: do REVA's two components transfer to ADE-150 (150 classes, NO background
class, dense labelling — the hardest regime for a background-sink mechanism)?

Cells (ADE-150 val, first 300 images, unified protocol, SCLIP + NACLIP):
- pixel plain (existing audit numbers used as reference where available)
- VABS-64 (vabs.py, full lexicon, tau=0.85, M=64 in two rounds of 32? -> use M=64
  single call) + matched randneg-64 control + SAM region pooling arms via
  probe_d1sam (pix_vabs / sam_reg_vabs / sam_reg_rand).

Disclosed leakage caveat: the VABS lexicon includes ADE-150 class names; the
tau_sim<0.85 filter removes words close to target classes, but residual leakage
is disclosed. Expectation (honest prior): VABS ~ 0 or negative here (no background
class to absorb into, as in Context-60); SAM region pooling should still give
>= +1 mIoU (mechanism independent of background). Criteria:
- E1: sam_reg_vabs - pix_vabs >= +1 -> region-evidence transfers.
- E2: record VABS effect vs plain and vs random honestly; any harm > 2 mIoU is
  written into the REVA limitations as a no-background failure regime.
