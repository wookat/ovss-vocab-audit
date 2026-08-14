"""Generate syn50 vocab variants for cocoobj/ctx60 using the frozen perturb.py rule."""
import json, random
import torch
from clip_seg import DenseCLIP
from perturb import synonyms, emb
import data

model = DenseCLIP("sclip", device="cuda" if torch.cuda.is_available() else "cpu")
for ds in ("cocoobj", "ctx60"):
    _, names, _ = data.DATASETS[ds]()
    base = [n.split(",")[0].strip() for n in names]
    E = emb(model, base)
    syn_choice = {}
    for i, n in enumerate(base):
        cands = synonyms(n)
        if not cands:
            continue
        ce = emb(model, cands)
        cos = (ce @ E[i]).tolist()
        valid = [(c, s) for c, s in zip(cands, cos) if 0.70 <= s <= 0.95]
        if valid:
            syn_choice[i] = sorted(valid, key=lambda t: -t[1])[0][0]
    idxs = sorted(syn_choice)
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        chosen = set(rng.sample(idxs, int(round(0.5 * len(idxs)))))
        vocab = [syn_choice[i] if i in chosen else n for i, n in enumerate(base)]
        json.dump(vocab, open(f"perturbed_vocabs/{ds}_syn50_s{seed}.json", "w"), indent=1)
    vocab = [syn_choice.get(i, n) for i, n in enumerate(base)]
    json.dump(vocab, open(f"perturbed_vocabs/{ds}_syn100_s0.json", "w"), indent=1)
    print(ds, "syn:", len(syn_choice), "/", len(base), flush=True)
