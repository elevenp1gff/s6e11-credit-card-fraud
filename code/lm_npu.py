"""Target-first language model member, in a compact row format.

The recipe (Chris Deotte, S6E10; GReaT) trains GPT-2 to write a row after its label, then scores a row under both
labels and applies Bayes. A natural-text row here is 224 GPT-2 tokens, of which only 40.5 carry a value, and the data
uses 1,372 of the 50,257 tokens. So this version keeps GPT-2's weights and its value tokens but:

  * one learned token per column replaces "column_name is " and ", "   (224 -> about 66 tokens per row)
  * the softmax covers the 1,372 used tokens plus END                  (LM head 36x smaller)
  * END is the target after a value's last token, so where a value stops still counts
  * column tokens are inputs, never targets: their order is our own shuffle and holds no information

usage (CUDA venv):  python lm_member.py check | bench | run [--folds 0,1,2,3,4,all]
"""
import argparse, ctypes, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAG = os.environ.get("TAG", "lm")
OUT = HERE / "out" / TAG
CK = HERE / "ckpt" / TAG
BOS = int(os.environ.get("BOS", 0))          # fixed first token, so the label is not at GPT-2's attention-sink position
LABEMB = int(os.environ.get("LABEMB", 0))    # add the label embedding at every position
NATURAL = int(os.environ.get("NATURAL", 0))  # the public text format: "is_fraud is True, col is value, ..."
PARITY = int(os.environ.get("PARITY", 0))    # train as the public script does: plain random batches, no warmup, loss on every token, bf16 logits
CACHE = HERE / ("cache_nat.npz" if NATURAL else "cache.npz")
FMT = f"b{BOS}_l{LABEMB}_n{NATURAL}"
for d in (OUT, CK):
    d.mkdir(parents=True, exist_ok=True)
try:
    import torch_npu
    dev = "npu:0" if torch.npu.is_available() else "cpu"
except Exception:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
log_candidates = dev
LMAX = 16 if NATURAL else 6                  # compact: column token + up to 5 value tokens
EPOCHS = int(os.environ.get("EPOCHS", 8))
BATCH = int(os.environ.get("BATCH", 32))
LR = float(os.environ.get("LR", 5e-5))
REP_EPOCHS = int(os.environ.get("REP_EPOCHS", 60))
REP_BATCH = int(os.environ.get("REP_BATCH", 16))
K_OOF = int(os.environ.get("K_OOF", 10))     # feature orders scored per held-out row
K_TEST = int(os.environ.get("K_TEST", 4))    # per test row, per fold model
K_REFIT = int(os.environ.get("K_REFIT", 10))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 0))   # fold 0: score the ab.py rows every this many epochs
SCORE_BS = 64 if NATURAL else 256
SEED = int(os.environ.get("SEED", 42))
pass

def _sync():
    try:
        if dev.startswith("npu"):
            torch.npu.synchronize()
        elif dev.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass

def _ac_dev():
    return "npu" if dev.startswith("npu") else "cuda"

def _empty_cache():
    try:
        if dev.startswith("npu"):
            torch.npu.empty_cache()
        elif dev.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass

def _max_mem():
    try:
        if dev.startswith("npu"):
            return torch.npu.max_memory_allocated()
        if dev.startswith("cuda"):
            return torch.cuda.max_memory_allocated()
    except Exception:
        pass
    return 0.0



def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


# ---------------------------------------------------------------- data
def build_cache():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(os.environ.get("GPT2_LOCAL","gpt2"))
    d = ROOT / "data"
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    tr, te = rd(d / "playground-series-s6e11/train.csv"), rd(d / "playground-series-s6e11/test.csv")
    og = rd(d / "original/credit_card_fraud_2026.csv")
    cols = [c for c in tr.columns if c not in ("id", "is_fraud")]
    allv = pd.concat([tr[cols], te[cols], og[cols]], ignore_index=True)
    used, per = set(), []
    pre = tok.encode("is_fraud is")
    lab = [tok.encode(" False")[0], tok.encode(" True")[0]]
    if NATURAL:
        used.update(pre + lab)
    for c in cols:
        codes, uniq = pd.factorize(allv[c], sort=False)
        head = tok.encode(",") + tok.encode(" " + c) + tok.encode(" is") if NATURAL else []
        ids = [head + tok.encode(" " + (str(v).strip() or "None")) for v in uniq]
        assert codes.min() >= 0 and max(map(len, ids)) <= (LMAX if NATURAL else LMAX - 1), c
        used.update(i for x in ids for i in x)
        per.append((codes, ids))
    vocab = np.array(sorted(used))
    remap = {g: i for i, g in enumerate(vocab)}
    V = len(vocab)
    COL0 = V + 4                              # V=END, V+1=PAD, V+2=label False, V+3=label True
    toks = np.full((len(allv), len(cols), LMAX), -1, np.int16)
    for j, (codes, ids) in enumerate(per):
        t = np.full((len(ids), LMAX), -1, np.int16)
        o = 0 if NATURAL else 1
        if not NATURAL:
            t[:, 0] = COL0 + j
        for i, x in enumerate(ids):
            t[i, o:o + len(x)] = [remap[g] for g in x]
        toks[:, j, :] = t[codes]
    y = np.r_[tr.is_fraud.astype(float).values, np.full(len(te), -1.0), og.is_fraud.astype(float).values].astype(np.int8)
    colids = np.full((len(cols), 8), -1, np.int64)
    for j, c in enumerate(cols):
        x = tok.encode(" " + c)[:8]
        colids[j, :len(x)] = x
    np.savez(CACHE, toks=toks, y=y, vocab=vocab, colids=colids, ntr=len(tr), nte=len(te), lab=np.array(lab),
             comma=np.array(tok.encode(",")), pre=np.array([remap[i] for i in pre] if NATURAL else [], np.int64),
             labn=np.array([remap[i] for i in lab] if NATURAL else [0, 0], np.int64))


def load_data():
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    g = dict(toks=z["toks"], y=z["y"].astype(np.int64), vocab=z["vocab"], colids=z["colids"], ntr=int(z["ntr"]),
             nte=int(z["nte"]), lab=z["lab"], comma=z["comma"])
    g["V"] = len(g["vocab"])
    g["END"], g["PAD"], g["LAB"], g["COL0"] = g["V"], g["V"] + 1, g["V"] + 2, g["V"] + 4
    g["NC"] = g["toks"].shape[1]
    g["BOS"] = g["COL0"] + g["NC"]
    g["NTOK"] = g["BOS"] + 1
    pre = [int(i) for i in z["pre"]] if "pre" in z.files else []
    g["PRE"] = pre if NATURAL else ([g["BOS"]] if BOS else [])
    g["LABID"] = [int(i) for i in z["labn"]] if NATURAL else [g["LAB"], g["LAB"] + 1]
    g["T"] = int((g["toks"] >= 0).sum((1, 2)).max()) + len(g["PRE"]) + 1      # + prefix and label
    if NATURAL:                                       # the replica writes the label as one more field, anywhere in the row
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(os.environ.get("GPT2_LOCAL","gpt2"))
        ids = [tok.encode(", is_fraud is " + s) for s in ("False", "True")]
        assert all(np.isin(i, g["vocab"]).all() for i in ids) and len(ids[0]) == len(ids[1]) <= LMAX
        g["LABF"] = np.full((2, LMAX), -1, np.int64)
        for r, i in enumerate(ids):
            g["LABF"][r, :len(i)] = np.searchsorted(g["vocab"], i)
        g["T"] += 2
    return g


class Rows:
    """Rows on the GPU; builds model inputs and targets without any host sync."""

    def __init__(self, g):
        self.t = torch.from_numpy(g["toks"]).to(dev)
        self.g, self.T = g, g["T"]
        self.P = len(g["PRE"])
        self.lens = (g["toks"] >= 0).sum((1, 2)) + self.P + 1
        self.pre = torch.tensor(g["PRE"], dtype=torch.long, device=dev)
        self.labid = torch.tensor(g["LABID"], dtype=torch.long, device=dev)
        self.labf = torch.as_tensor(g["LABF"], device=dev) if NATURAL else None

    def perm(self, n, gen):
        return torch.rand(n, self.g["NC"], device=dev, generator=gen).argsort(1)

    def batches(self, rows, bs, rs, group=64):
        """Random batches of similar length, so each batch pads only to its own longest row."""
        group = 1 if PARITY else group
        o = rs.permutation(rows)[:len(rows) // bs * bs]
        for c in range(0, len(o), bs * group):
            s = o[c:c + bs * group]
            o[c:c + bs * group] = s[np.argsort(self.lens[s], kind="stable")]
        b = o.reshape(-1, bs)[rs.permutation(len(o) // bs)]
        return torch.as_tensor(b, device=dev), self.lens[b].max(1).tolist()

    def assemble_any(self, idx, lab, gen, T):
        """A replica row as be_great writes it: every field in a fresh order, the label among them, loss on all tokens."""
        g = self.g
        B = idx.numel()
        perm = torch.rand(B, g["NC"] + 1, device=dev, generator=gen).argsort(1)
        f = torch.cat([self.t[idx].long(), self.labf[lab][:, None, :]], 1)
        seg = f.gather(1, perm[:, :, None].expand(-1, -1, LMAX)).reshape(B, -1)
        valid = seg >= 0
        pos = torch.where(valid, valid.cumsum(1) - 1, torch.full_like(seg, T + 1))
        buf = torch.full((B, T + 2), g["PAD"], dtype=torch.long, device=dev)
        buf.scatter_(1, pos, seg.clamp_min(0))
        buf = buf[:, :T + 1]
        cur, nxt = buf[:, :-1], buf[:, 1:]
        tgt = torch.where(nxt < g["V"], nxt, torch.full_like(nxt, g["END"]))
        tgt = torch.where(cur == g["PAD"], torch.full_like(tgt, -100), tgt)
        return cur.contiguous(), tgt

    def assemble(self, idx, lab, perm, T=None, full=False):
        g, T = self.g, T or self.T
        B = idx.numel()
        seg = self.t[idx].long().gather(1, perm[:, :, None].expand(-1, -1, LMAX)).reshape(B, -1)
        valid = seg >= 0
        pos = torch.where(valid, valid.cumsum(1) + self.P, torch.full_like(seg, T + 1))   # T+1 is a dump column
        buf = torch.full((B, T + 2), g["PAD"], dtype=torch.long, device=dev)
        buf.scatter_(1, pos, seg.clamp_min(0))
        buf = buf[:, :T + 1]
        buf[:, :self.P] = self.pre
        buf[:, self.P] = self.labid[lab]
        cur, nxt = buf[:, :-1], buf[:, 1:]
        tgt = torch.where(nxt < g["V"], nxt, torch.full_like(nxt, g["END"]))     # a column token or pad ends the value
        tgt = torch.where(cur == g["PAD"], torch.full_like(tgt, -100), tgt)
        if not full:
            tgt[:, :self.P + (0 if NATURAL else 1)] = -100                       # score only what follows the label
        return cur.contiguous(), tgt


# ---------------------------------------------------------------- model (GPT-2 small, same maths as the HF module)
class Block(nn.Module):
    def __init__(s, d, nh, p):
        super().__init__()
        s.ln_1, s.ln_2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.fc, s.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        s.nh, s.p, s.drop = nh, p, nn.Dropout(p)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln_1(x)).split(D, 2)
        sh = lambda t: t.view(B, T, s.nh, D // s.nh).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), dropout_p=s.p if s.training else 0.0, is_causal=True)
        x = x + s.drop(s.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + s.drop(s.out(F.gelu(s.fc(s.ln_2(x)), approximate="tanh")))


class LM(nn.Module):
    def __init__(s, ntok, nout, T, nl=12, nh=12, d=768, p=0.1):
        super().__init__()
        s.wte, s.wpe = nn.Embedding(ntok, d), nn.Embedding(T, d)
        s.h = nn.ModuleList(Block(d, nh, p) for _ in range(nl))
        s.ln_f, s.drop, s.nout = nn.LayerNorm(d, eps=1e-5), nn.Dropout(p), nout

    def hidden(s, x, lab=None):
        e = s.wte(x) + s.wpe.weight[:x.shape[1]]
        h = s.drop(e if lab is None else e + s.wte(lab)[:, None, :])
        for b in s.h:
            h = b(h)
        return s.ln_f(h)

    def forward(s, x, lab=None):
        if PARITY and s.training:                   # under autocast: bf16 logits, as in the public training step
            return F.linear(s.hidden(x, lab), s.wte.weight[:s.nout])
        h = s.hidden(x, lab).float()                # logits reach about 100: bf16 would round them to 0.5
        with torch.autocast(_ac_dev(), enabled=False):
            return h @ s.wte.weight[:s.nout].T


def copy_blocks(m, sd):
    P = "transformer."
    with torch.no_grad():
        for i, b in enumerate(m.h):
            q = f"{P}h.{i}."
            for name, key in (("ln_1", "ln_1"), ("ln_2", "ln_2")):
                getattr(b, name).weight.copy_(sd[q + key + ".weight"])
                getattr(b, name).bias.copy_(sd[q + key + ".bias"])
            for name, key in (("qkv", "attn.c_attn"), ("proj", "attn.c_proj"), ("fc", "mlp.c_fc"), ("out", "mlp.c_proj")):
                getattr(b, name).weight.copy_(sd[q + key + ".weight"].T)        # HF Conv1D stores [in, out]
                getattr(b, name).bias.copy_(sd[q + key + ".bias"])
        m.ln_f.weight.copy_(sd[P + "ln_f.weight"])
        m.ln_f.bias.copy_(sd[P + "ln_f.bias"])


def from_gpt2(g):
    from transformers import GPT2LMHeadModel
    sd = GPT2LMHeadModel.from_pretrained(os.environ.get("GPT2_LOCAL","gpt2")).state_dict()
    m = LM(g["NTOK"], g["V"] + 1, g["T"])
    copy_blocks(m, sd)
    w = sd["transformer.wte.weight"]
    with torch.no_grad():
        m.wpe.weight.copy_(sd["transformer.wpe.weight"][:g["T"]])
        m.wte.weight[:g["V"]] = w[torch.from_numpy(g["vocab"])]
        m.wte.weight[g["END"]] = w[50256 if NATURAL else int(g["comma"][0])]
        m.wte.weight[g["BOS"]] = w[50256]
        m.wte.weight[g["PAD"]] = w.mean(0)
        m.wte.weight[g["LAB"]:g["LAB"] + 2] = w[torch.from_numpy(g["lab"])]
        for j, r in enumerate(g["colids"]):
            m.wte.weight[g["COL0"] + j] = w[torch.from_numpy(r[r >= 0])].mean(0)
    return m.to(dev)


# ---------------------------------------------------------------- train / score
def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(m, R, ytr, rows, epochs, bs, lr, name, seed, max_steps=None, anywhere=False, hook=None):
    per = len(rows) // bs
    total = per * epochs
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0, fused=True)
    ck, start = CK / f"{name}.pt", 0
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=True)
        m.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        start = int(s["epoch"]) + 1
        log(f"{name}: resumed after epoch {start}")
    gen = torch.Generator(device=dev)
    m.train()
    t0, seen, run = time.time(), 0, torch.zeros((), device=dev)
    for ep in range(start, epochs):
        run.zero_()
        gen.manual_seed(seed * 1000 + ep)
        bt, tb = R.batches(rows, bs, np.random.RandomState((seed * 1000 + ep) % 2**31))
        for s in range(per):
            step = ep * per + s
            for grp in opt.param_groups:
                grp["lr"] = lr * (1.0 if PARITY else min(1.0, (step + 1) / 100)) * max(0.0, 1 - step / total)
            idx = bt[s]
            if anywhere:
                inp, tgt = R.assemble_any(idx, ytr[idx], gen, tb[s] + 2)
            else:
                inp, tgt = R.assemble(idx, ytr[idx], R.perm(bs, gen), tb[s], full=bool(PARITY))
            with torch.autocast(_ac_dev(), dtype=torch.bfloat16):
                lg = m(inp, R.labid[ytr[idx]] if LABEMB else None)
            loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.detach()
            seen += bs
            if (s + 1) % 2000 == 0:
                log(f"{name} ep {ep + 1}/{epochs} step {s + 1}/{per} loss {run.item() / 2000:.4f} "
                    f"{seen / (time.time() - t0):.0f} rows/s")
                run.zero_()
            if max_steps and s + 1 >= max_steps:
                (_sync())
                return seen / (time.time() - t0)
        save_atomic({"model": m.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ck)
        log(f"{name}: epoch {ep + 1}/{epochs} done, {(time.time() - t0) / 60:.1f} min")
        if hook:
            hook(m, ep)
    return None


@torch.no_grad()
def score(m, R, rows, K, seed, bs=256, dtype=torch.float16):
    """Log-likelihood of each row under False and True, for K shuffled feature orders: float32 [n, K, 2]."""
    m.eval()
    _empty_cache()                         # training's cached blocks push scoring past 8 GB into shared memory: 8x slower
    srt = np.argsort(R.lens[rows], kind="stable")
    ls = R.lens[rows][srt]
    rows_t = torch.as_tensor(rows[srt], device=dev)
    out = torch.zeros((len(rows), K, 2), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for k in range(K):
        for s in range(0, len(rows), bs):
            idx = rows_t[s:s + bs]
            n = idx.numel()
            perm = R.perm(n, gen)
            inp, tgt = R.assemble(idx, torch.zeros(n, dtype=torch.long, device=dev), perm, int(ls[s + n - 1]))
            both = torch.cat([inp, inp])
            both[n:, R.P] = R.labid[1]
            lab2 = R.labid.repeat_interleave(n) if LABEMB else None
            with torch.autocast(_ac_dev(), dtype=dtype):
                lg = m(both, lab2)
            nll = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.repeat(2, 1).reshape(-1), ignore_index=-100,
                                  reduction="none").view(2 * n, -1).sum(1)
            out[s:s + n, k, 0] = -nll[:n]
            out[s:s + n, k, 1] = -nll[n:]
    res = np.empty((len(rows), K, 2), np.float32)
    res[srt] = out.cpu().numpy()
    del out
    _empty_cache()
    return res


def bayes(ll, prior):
    """Log-sum-exp over orders per label, difference, plus the prior log-odds."""
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1 - prior))


def replica(g, R, ydev):
    """The generator's start: the original 20k rows, written label first."""
    p = CK.parent / f"replica_{FMT}.pt"              # shared by every run of this format
    if p.exists():
        m = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
        m.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
        return m
    m = from_gpt2(g)
    rows = np.arange(g["ntr"] + g["nte"], len(g["y"]))
    train(m, R, ydev, rows, REP_EPOCHS, REP_BATCH, LR, "replica", SEED, anywhere=bool(NATURAL))
    save_atomic(m.state_dict(), p)
    (CK / "replica.pt").unlink(missing_ok=True)
    return m


def run(g, which):
    R = Rows(g)
    y = g["y"]
    ydev = torch.as_tensor(np.maximum(y, 0), device=dev)
    ntr, nte = g["ntr"], g["nte"]
    test = np.arange(ntr, ntr + nte)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    t0 = time.time()
    replica(g, R, ydev)
    log(f"replica ready, {(time.time() - t0) / 60:.1f} min")
    for f in which:
        done = OUT / f"test_{f}.npy"
        if done.exists():
            log(f"fold {f}: already scored")
            continue
        t1 = time.time()
        a = np.arange(ntr) if f == "all" else folds[int(f)][0]
        m = replica(g, R, ydev)
        hook = None
        if EVAL_EVERY and f == "0":
            rs = np.random.RandomState(0)             # the same rows as ab.py and tune.py
            rs.choice(folds[0][0], 280000, replace=False)
            b0 = folds[0][1]
            ev = np.r_[b0[y[b0] == 1], rs.choice(b0[y[b0] == 0], 40000, replace=False)]

            def hook(mm, ep):
                if (ep + 1) % EVAL_EVERY:
                    return
                ll = score(mm, R, ev, 2, 7, bs=SCORE_BS)
                np.save(OUT / f"ep{ep + 1}.npy", ll)
                log(f"fold 0 epoch {ep + 1}: eval AUC {roc_auc_score(y[ev], (ll[:, :, 1] - ll[:, :, 0]).mean(1)):.6f}")
                mm.train()
        train(m, R, ydev, a, EPOCHS, BATCH, LR, f"fold_{f}", SEED + (9 if f == "all" else int(f)), hook=hook)
        t2 = time.time()
        prior = y[a].mean()
        if f != "all":
            b = folds[int(f)][1]
            ll = score(m, R, b, K_OOF, SEED + 100 + int(f), bs=SCORE_BS)
            np.save(OUT / f"oof_{f}.npy", ll)
            z = bayes(ll, prior)
            log(f"fold {f}: AUC {roc_auc_score(y[b], z):.6f} (one order {roc_auc_score(y[b], ll[:, 0, 1] - ll[:, 0, 0]):.6f})")
        lt = score(m, R, test, K_REFIT if f == "all" else K_TEST, SEED + 200 + (9 if f == "all" else int(f)), bs=SCORE_BS)
        np.save(done, lt)
        log(f"fold {f}: train {(t2 - t1) / 60:.1f} min, score {(time.time() - t2) / 60:.1f} min")
        del m                                       # the fold checkpoint stays: tune.py starts from it
        _empty_cache()
    log("all requested folds finished")


# ---------------------------------------------------------------- checks
def check(g):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(os.environ.get("GPT2_LOCAL","gpt2"))
    hf = GPT2LMHeadModel.from_pretrained(os.environ.get("GPT2_LOCAL","gpt2"), attn_implementation="eager").eval()
    m = LM(50257, 50257, 1024, p=0.0).eval()
    copy_blocks(m, hf.state_dict())
    with torch.no_grad():
        m.wte.weight.copy_(hf.state_dict()["transformer.wte.weight"])
        m.wpe.weight.copy_(hf.state_dict()["transformer.wpe.weight"])
        ids = torch.tensor([tok.encode("is_fraud is True, amount_usd is 61.16, card_type is Visa")])
        d = (m(ids) - hf(ids).logits).abs().max().item()
    print("max logit difference vs Hugging Face GPT-2:", d)
    assert d < 1e-2
    R = Rows(g)
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = torch.tensor([0, 1], device=dev)
    inp, tgt = R.assemble(idx, torch.tensor([0, 1], device=dev), R.perm(2, gen))
    names = lambda t: ("END" if t == g["END"] else "PAD" if t == g["PAD"] else "F" if t == g["LAB"] else
                       "T" if t == g["LAB"] + 1 else f"<c{t - g['COL0']}>" if t >= g["COL0"] else
                       "-" if t < 0 else tok.decode([int(g["vocab"][t])]))
    print("T", g["T"], "V", g["V"], "NTOK", g["NTOK"])
    for r in range(2):
        print(" ".join(f"{names(int(a))}>{names(int(b))}" for a, b in zip(inp[r].tolist(), tgt[r].tolist()))[:900])
    real = (g["toks"][:g["ntr"]] >= 0).sum((1, 2))
    print("tokens per row: mean", real.mean() + 1, "max", real.max() + 1)
    # every value token of a row must come back as a target exactly once, plus one END per field
    n_t = int((tgt[0] >= 0).sum())
    print("targets row 0:", n_t, "expected", int(real[0]), "END count", int((tgt[0] == g["END"]).sum()))
    assert n_t == int(real[0]) and int((tgt[0] == g["END"]).sum()) == g["NC"]
    mm = from_gpt2(g).eval()
    a = np.arange(2048)
    l16 = score(mm, R, a, 2, 1, dtype=torch.float16)
    pass
    l32 = score(mm, R, a, 2, 1, dtype=torch.float32)
    d = np.abs((l16[..., 1] - l16[..., 0]) - (l32[..., 1] - l32[..., 0]))
    print("fp16 vs exact fp32 scoring, |LLR difference| mean", d.mean(), "max", d.max(), "LLR sd", (l32[..., 1] - l32[..., 0]).std())
    # rows must come back in their own place after the length sort
    one = score(mm, R, a[[7]], 1, 1)
    print("row 7 alone", one[0, 0], "in batch", l16[7, 0], "(orders differ, so only the size must match)")


def bench(g):
    R = Rows(g)
    ydev = torch.as_tensor(np.maximum(g["y"], 0), device=dev)
    rows = np.arange(g["ntr"])
    for bs in (32, 64):
        m = from_gpt2(g)
        r = train(m, R, ydev, rows, 1, bs, LR, f"bench{bs}", 1, max_steps=150)
        print(f"train batch {bs}: {r:.0f} rows/s, peak {_max_mem() / 2**30:.2f} GB", flush=True)
        (CK / f"bench{bs}.pt").unlink(missing_ok=True)
    a = np.arange(40000)
    for bs in (256, 512):
        t = time.time()
        score(m, R, a, 1, 1, bs=bs)
        (_sync())
        print(f"score batch {bs}: {len(a) / (time.time() - t):.0f} rows/s (both labels, one order)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "bench", "run"])
    ap.add_argument("--folds", default="0,1,2,3,4,all")
    a = ap.parse_args()
# [linux] removed: ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)      # keep the laptop awake while this process lives
    torch.manual_seed(SEED)
    g = load_data()
    if a.mode == "check":
        check(g)
    elif a.mode == "bench":
        bench(g)
    else:
        log(f"start: epochs {EPOCHS} batch {BATCH} lr {LR} K {K_OOF}/{K_TEST}/{K_REFIT} seed {SEED} T {g['T']} V {g['V']}")
        run(g, a.folds.split(","))
