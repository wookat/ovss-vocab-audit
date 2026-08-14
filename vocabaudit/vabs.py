"""VABS: Vocabulary-Adaptive Background Synthesis (training-free, text-only).

Given any user vocabulary, automatically select M negative words from a
vocabulary-independent lexicon D (COCO-Stuff-171 ∪ ADE20K-150 ∪ WordNet concrete
nouns), filtered for safety (max cosine to any target class < tau_sim), chosen by
facility-location greedy coverage with redundancy penalty. Negatives are folded into
the background class via the sub-query (comma) convention of the eval harness.

Usage:
  python vabs.py --vocab-file perturbed_vocabs/voc21_plain.json --out perturbed_vocabs/voc21_plain_vabs.json
  python vabs.py --dataset voc21 --out perturbed_vocabs/voc21_official_vabs.json   # official names
  python vabs.py --vocab-file ... --random --seed 0 --out ...                      # random-negatives control
"""
import json, argparse, random
import torch
import torch.nn.functional as F
from clip_seg import DenseCLIP
from eval_seg import TEMPLATES
import data

EXCLUDE = {"background", "unknown", "other"}


def build_lexicon(kind="full"):
    words = set()
    if kind == "ade":
        src = list(data.ade150()[1])
    else:
        src = list(data.coco171()[1]) + list(data.ade150()[1])
    for nm in src:
        for s in nm.split(","):
            s = s.strip().lower()
            if s and s not in EXCLUDE:
                words.add(s)
    if kind in ("scene", "ade"):
        return sorted(words)
    try:
        from nltk.corpus import wordnet as wn
        LEX = {"noun.object", "noun.artifact", "noun.substance", "noun.plant",
               "noun.animal", "noun.body", "noun.food", "noun.location", "noun.phenomenon"}
        for syn in wn.all_synsets("n"):
            if syn.lexname() not in LEX:
                continue
            for lemma in syn.lemmas():
                if lemma.count() >= 3 and "_" not in lemma.name():
                    w = lemma.name().lower()
                    if w not in EXCLUDE and len(w) > 2:
                        words.add(w)
    except Exception as e:
        print("wordnet unavailable:", e)
    return sorted(words)


@torch.no_grad()
def embed(model, words, bs=64):
    out = []
    for i in range(0, len(words), bs):
        chunk = words[i:i + bs]
        embs = []
        for w in chunk:
            e = model.encode_text_raw([t.format(w) for t in TEMPLATES])
            embs.append(F.normalize(e.mean(0), dim=-1))
        out.append(torch.stack(embs))
    return torch.cat(out)


@torch.no_grad()
def steal_rates(model, guard_ds, T_V, cand_emb, offset=300, n_imgs=100, conf=0.0, scale=40.0):
    """For each candidate negative, max over target classes of the fraction of
    high-confidence target patches (targets-only softmax prob >= conf) that the
    candidate would steal (sim(n) > sim(t)). Images only, no GT."""
    from PIL import Image
    from eval_seg import resize_short, to_tensor
    samples = data.DATASETS[guard_ds]()[0][offset:offset + n_imgs]
    K = T_V.shape[0]
    steal = torch.zeros(cand_emb.shape[0], K, device=model.device)
    count = torch.zeros(K, device=model.device)
    Tv = T_V.to(model.device)
    Ce = cand_emb.to(model.device)
    for img_path, _, _ in samples:
        img = Image.open(img_path).convert("RGB")
        img_r, _ = resize_short(img, 336)
        t = to_tensor(img_r, model.device)
        _, _, H, W = t.shape
        y, x = (H - 224) // 2, (W - 224) // 2
        feat, _ = model.encode_dense(t[:, :, y:y + 224, x:x + 224])
        feat = F.normalize(feat[0].float(), dim=-1)  # (P, D)
        sim_t = feat @ Tv.T                          # (P, K)
        prob = (scale * sim_t).softmax(-1)
        pmax, assign = prob.max(-1)
        hi = pmax >= conf
        if hi.sum() == 0:
            continue
        sim_c = feat[hi] @ Ce.T                      # (Ph, n_c)
        best_t = sim_t[hi].gather(1, assign[hi].unsqueeze(1))  # (Ph,1)
        flips = (sim_c > best_t)                     # (Ph, n_c)
        for k in range(K):
            mk = assign[hi] == k
            if mk.any():
                steal[:, k] += flips[mk].float().sum(0)
                count[k] += mk.sum()
    return (steal / count.clamp_min(1).unsqueeze(0)).cpu()  # (n_c, K)


def select_negatives(T_D, lexicon, T_V, M=32, tau_sim=0.85, lam=0.5):
    sim_to_targets = (T_D @ T_V.T).max(1).values
    keep = sim_to_targets < tau_sim
    C = T_D[keep]
    cwords = [w for w, k in zip(lexicon, keep.tolist()) if k]
    S = C @ C.T  # (n,n) candidate-candidate sim
    n = S.shape[0]
    covered = torch.zeros(n, device=S.device)
    chosen = []
    for _ in range(M):
        gain = (S - covered.unsqueeze(0)).clamp_min(0).mean(1)
        if chosen:
            idx = torch.tensor(chosen, device=S.device, dtype=torch.long)
            red = S[:, idx].max(1).values
            gain = gain - lam * red.clamp_min(0) * gain.mean()
            gain[idx] = -1e9
        d = int(gain.argmax())
        chosen.append(d)
        covered = torch.maximum(covered, S[d])
    return [cwords[i] for i in chosen], cwords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-file", default=None)
    ap.add_argument("--dataset", default="voc21")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--tau-sim", type=float, default=0.85)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lexicon", default="full", choices=["full", "scene", "ade"])
    ap.add_argument("--guard", type=float, default=None, help="visual steal guard gamma (drop)")
    ap.add_argument("--reassign", type=float, default=None,
                    help="steal threshold: negatives stealing more than this from a target\n"
                         "class become that class's sub-query instead of background")
    ap.add_argument("--guard-dataset", default="voc21")
    ap.add_argument("--guard-offset", type=int, default=300)
    ap.add_argument("--guard-n", type=int, default=100)
    ap.add_argument("--random", action="store_true", help="random-negatives control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.vocab_file:
        names = json.load(open(a.vocab_file))
    else:
        names = data.DATASETS[a.dataset]()[1]

    model = DenseCLIP("sclip", device="cuda")
    lexicon = build_lexicon(a.lexicon)
    print(f"lexicon size {len(lexicon)}")
    T_D = embed(model, lexicon)
    targets = [s.strip() for n in names for s in n.split(",") if s.strip().lower() not in EXCLUDE]
    T_V = embed(model, targets)

    sim_to_targets = (T_D @ T_V.T).max(1).values
    keep = sim_to_targets < a.tau_sim
    cwords_all = [w for w, k in zip(lexicon, keep.tolist()) if k]
    if a.guard is not None:
        cand_emb = T_D[keep]
        rates = steal_rates(model, a.guard_dataset, T_V, cand_emb,
                            offset=a.guard_offset, n_imgs=a.guard_n).max(1).values
        safe = rates <= a.guard
        dropped = [w for w, s in zip(cwords_all, safe.tolist()) if not s]
        print(f"guard gamma={a.guard}: dropped {len(dropped)}/{len(cwords_all)}:", dropped[:40])
        lex_g = [w for w, s in zip(cwords_all, safe.tolist()) if s]
        T_Dg = cand_emb[safe]
    else:
        lex_g, T_Dg = cwords_all, T_D[keep]
    if a.random:
        rng = random.Random(a.seed)
        negs = rng.sample(lex_g, a.M)
    else:
        negs, _ = select_negatives(T_Dg, lex_g, T_V, a.M, 1.1, a.lam)
    print("negatives:", negs)

    attach = {}  # class index (into names) -> extra sub-queries
    if a.reassign is not None:
        E_N = embed(model, negs)
        r = steal_rates(model, a.guard_dataset, T_V, E_N,
                        offset=a.guard_offset, n_imgs=a.guard_n)  # (M, Kq)
        # map query index -> names index
        q2n = []
        for ni, n in enumerate(names):
            for s in n.split(","):
                if s.strip().lower() not in EXCLUDE:
                    q2n.append(ni)
        mx, am = r.max(1)
        keep_negs = []
        for i, w in enumerate(negs):
            if mx[i] > a.reassign:
                attach.setdefault(q2n[am[i]], []).append(w)
            else:
                keep_negs.append(w)
        print("reassigned:", {names[k].split(",")[0]: v for k, v in attach.items()})
        negs = keep_negs

    out_names, folded = [], False
    for ni, n in enumerate(names):
        subs = [s.strip() for s in n.split(",")] + attach.get(ni, [])
        if any(s.lower() == "background" for s in subs):
            out_names.append(", ".join(subs + negs))
            folded = True
        else:
            out_names.append(", ".join(subs))
    if not folded:
        out_names.append(", ".join(["background"] + negs))
    json.dump(out_names, open(a.out, "w"), indent=1)
    meta = {"M": a.M, "tau_sim": a.tau_sim, "lam": a.lam, "lexicon": a.lexicon,
            "guard": a.guard, "reassign": a.reassign,
            "attach": {names[k]: v for k, v in attach.items()}, "random": a.random,
            "seed": a.seed, "lexicon_size": len(lexicon), "negatives": negs,
            "source_vocab": a.vocab_file or a.dataset}
    json.dump(meta, open(a.out.replace(".json", "_meta.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
