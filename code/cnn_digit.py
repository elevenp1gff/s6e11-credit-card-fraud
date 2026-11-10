"""cnnv5 - the decimal-digit CNN, built the way the reference describes it.

WHY v1/v2/v3 PLATEAUED AT 0.943-0.945
A LightGBM on the raw 19 columns scores 0.9506 OOF on the identical split
(out/score_gbr_e.json), so neither the data nor the split is the ceiling. Five
concrete defects in my own pipeline explain the gap:

 1. THE CONV RAN OVER THE WRONG AXIS. "A shared CNN reads decimal digits for 13
    numeric fields" means one CNN applied to each field's digit string, i.e. a
    convolution along the DIGIT axis, shared across fields, followed by a
    permutation-invariant pool over FIELDS. v3 instead put the 13 fields on the
    conv axis and convolved ACROSS them - but field order is arbitrary
    (amount, hours, txn_count, distance, ...), so that layer learned a bias over a
    meaningless ordering, and no per-field digit pattern was ever modelled.
    v4/cnnv5: digits (B,F,D) -> shared Conv1d over D -> pool over D -> DeepSets
    mean + projected max over F.

 2. THE LR SCHEDULE NEVER COMPLETED. Every earlier run built
    OneCycleLR(total_steps = 60 epochs) but early-stopped at epoch 11-19 (the
    largest epoch ever logged is 21). OneCycle reaches peak LR at 30% of budget,
    so stopping fired near peak LR and the net was never evaluated at small LR.
    Fix: cosine schedule, always runs all epochs, no early stop.

 3. SIX BOOLEAN FEATURES WERE DISCARDED - only their sum reached the network.
    Fix: all six plus the sum are inputs (20 numeric inputs).

 4. N_DEC=3 WAS ALREADY PADDING. Verified on train.csv: every numeric column has
    at most 2 decimals, so 3 decimal digits inject a constant zero and 4-5 add
    more zeros. Fix: W_INT=5, W_DEC=2 exactly.

 5. LEADING-ZERO PADDING. A 5-digit integer window padded with literal 0 digits
    tells the shared CNN that "small number" and "number with leading zeros" are
    the same thing. Fix: a PAD token (id 10) for positions left of the most
    significant digit.

Also: raw decimal digits (--enc raw) vs v3's integer-rescaled encoding
(--enc norm) as a control arm, per-fold QuantileTransformer instead of global
mean/std, and predictions averaged over the final/top epochs.

NOTE ON THIS BUILD: nn.Linear(n,1) raises "could not create a primitive
descriptor for a matmul primitive" on CPU in this torch 2.1.0 aarch64 env
(width>=2 is fine). Class Out keeps a CPU fallback working.
"""
import os, sys, json, time, argparse, copy, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = os.environ.get("S6E11_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
        "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUM = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
       "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
       "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score",
       "prior_disputes"]
W_INT, W_DEC = 5, 2
NDIG = W_INT + W_DEC
FIELDS = len(NUM)
PAD = 10


def _digits(v, pad_blank):
    """Decimal digits of v as int8, integer part first (W_INT) then decimals."""
    ip = np.floor(v + 1e-9).astype("int64")
    fr = np.clip(np.rint((v - ip) * (10 ** W_DEC)).astype("int64"), 0, 10 ** W_DEC - 1)
    ipc = np.clip(ip, 0, 10 ** W_INT - 1)
    d = np.zeros((len(v), NDIG), dtype=np.int8)
    for k in range(W_INT):
        d[:, k] = (ipc // (10 ** (W_INT - 1 - k))) % 10
    for k in range(W_DEC):
        d[:, W_INT + k] = (fr // (10 ** (W_DEC - 1 - k))) % 10
    if pad_blank:
        nlead = (W_INT - np.maximum(1, np.floor(np.log10(np.maximum(ipc, 1))).astype(int) + 1))
        for k in range(W_INT):
            d[nlead > k, k] = PAD
    return d


def build_cache():
    t0 = time.time()
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    assert tr["is_fraud"].notna().all()
    y = tr["is_fraud"].astype(int).values
    n_tr = len(tr)
    full = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)

    Dg = np.zeros((len(full), FIELDS, NDIG), dtype=np.int8)   # raw digits, PAD for leading
    Dz = np.zeros((len(full), FIELDS, NDIG), dtype=np.int8)   # raw digits, literal zeros
    Dn = np.zeros((len(full), FIELDS, NDIG), dtype=np.int8)   # v3 integer-rescaled
    for j, c in enumerate(NUM):
        v = full[c].values.astype("float64")
        assert v.min() >= 0, c
        Dg[:, j, :] = _digits(v, True)
        Dz[:, j, :] = _digits(v, False)
        ip = np.floor(v + 1e-9).astype("int64")
        mx = max(1, int(ip.max()))
        sc = (ip * (10 ** W_INT) // (mx + 1)).astype("int64")
        for k in range(W_INT):
            Dn[:, j, k] = (sc // (10 ** (W_INT - 1 - k))) % 10
        Dn[:, j, W_INT:] = Dz[:, j, W_INT:]

    Xn = np.empty((len(full), FIELDS + len(BOOL) + 1), dtype="float32")
    for j, c in enumerate(NUM):
        Xn[:, j] = full[c].values.astype("float32")
    for j, c in enumerate(BOOL):
        Xn[:, FIELDS + j] = full[c].values.astype("float32")
    Xn[:, FIELDS + len(BOOL)] = full[BOOL].sum(axis=1).values.astype("float32")

    C = np.zeros((len(full), len(CAT)), dtype="int64")
    for j, c in enumerate(CAT):
        C[:, j] = pd.Categorical(full[c]).codes

    for name, arr in [("Dg", Dg), ("Dz", Dz), ("Dn", Dn), ("Xn", Xn), ("C", C), ("y", y)]:
        np.save(os.path.join(OUT, f"v5_{name}.npy"), arr)
    with open(os.path.join(OUT, "v5_meta.json"), "w") as f:
        json.dump({"n_tr": int(n_tr), "n_te": int(len(te)), "card": int(C.max()) + 1,
                   "W_INT": W_INT, "W_DEC": W_DEC, "fields": FIELDS}, f)
    print(f"cache built {Dg.shape} in {time.time()-t0:.0f}s card={C.max()+1}", flush=True)


def load_cache(enc):
    mm = lambda n: np.load(os.path.join(OUT, f"v5_{n}.npy"), mmap_mode="r")
    key = {"raw": "Dg", "zeros": "Dz", "norm": "Dn"}[enc]
    meta = json.load(open(os.path.join(OUT, "v5_meta.json")))
    return mm(key), mm("Xn"), mm("C"), mm("y"), meta


class Out(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.lin = nn.Linear(nin, 2)

    def forward(self, x):
        return self.lin(x)[:, 0]


class CNN(nn.Module):
    """Shared CNN over each field's decimal digits, DeepSets pool over fields."""

    def __init__(self, n_num, n_cat, card, dim=48, emb=16, p=0.15, ndig=NDIG,
                 arch="deepsets", nfield=FIELDS):
        super().__init__()
        self.arch = arch
        self.emb = nn.Embedding(PAD + 1, 8, padding_idx=PAD)
        self.c1 = nn.Conv1d(8, dim, 3, padding=1)
        self.b1 = nn.BatchNorm1d(dim)
        self.c2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.b2 = nn.BatchNorm1d(dim)
        self.cat_emb = nn.ModuleList([nn.Embedding(card, emb) for _ in range(n_cat)])
        self.pool_proj = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.SiLU())
        in_dim = (nfield * 2 * dim if arch == "flat" else 2 * dim) \
            + 2 * dim + n_num + n_cat * emb
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(p),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(p),
            Out(256))

    def forward(self, d, xn, c):
        B, F, Dg = d.shape
        e = self.emb(d)                              # (B,F,D,8)
        x = e.permute(0, 1, 3, 2).reshape(B * F, 8, Dg)
        h = Fn.silu(self.b1(self.c1(x)))
        h = Fn.silu(self.b2(self.c2(h)) + h)
        per = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1).reshape(B, F, -1)
        if self.arch == "flat":
            # Keep WHICH field each digit pattern came from. A pure permutation-
            # invariant pool cannot tell amount_usd digits from customer_age digits,
            # and the shared CNN therefore has no way to specialise per field.
            resid = self.pool_proj(per.amax(dim=1))  # DeepSets residual from max
            parts = [per.reshape(B, -1), resid, xn]
        else:
            mean_f = per.mean(dim=1)                 # DeepSets mean over fields
            resid = self.pool_proj(per.amax(dim=1))  # DeepSets residual from max
            parts = [mean_f, resid, xn]
        for j, m in enumerate(self.cat_emb):
            parts.append(m(c[:, j]))
        return self.head(torch.cat(parts, 1)).squeeze(-1)


def make_loss(kind):
    """AUC is a ranking metric on a 1.67%-positive set; plain BCE spends almost all
    gradient on the negative majority. focal/pos variants reweight the rare class."""
    if kind == "bce":
        return nn.BCEWithLogitsLoss()

    def focal(logit, target):
        p = torch.sigmoid(logit)
        ce = Fn.binary_cross_entropy_with_logits(logit, target, reduction="none")
        pt = p * target + (1 - p) * (1 - target)
        return ((1 - pt) ** 2 * ce).mean()
    if kind == "focal":
        return focal
    if kind == "pos":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    raise ValueError(kind)


def get_device():
    want = os.environ.get("DEV", "npu")
    if want.startswith("npu"):
        try:
            import torch_npu
            if torch.npu.is_available():
                torch.npu.set_device(want)
                return torch.device(want)
        except Exception as e:
            print("npu unavailable:", str(e)[:120], flush=True)
    return torch.device("cpu")


def _predict(m, d, x, c, dev, bs=16384):
    m.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(d), bs):
            o = m(d[i:i + bs], x[i:i + bs], c[i:i + bs])
            outs.append(torch.sigmoid(o).float().cpu().numpy())
    return np.concatenate(outs).astype("float64")


def run_fold(fold_idx, args):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import QuantileTransformer
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    t0 = time.time()
    D, Xn, C, y, meta = load_cache(args.enc)
    n_tr, card = meta["n_tr"], meta["card"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_tr), y))
    itr, iva = folds[fold_idx]

    Dtr = np.ascontiguousarray(D[:n_tr][itr]); Dva = np.ascontiguousarray(D[:n_tr][iva])
    Dte = np.ascontiguousarray(D[n_tr:])
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200000, random_state=0).fit(np.asarray(Xn[:n_tr][itr]))
    Xtr = qt.transform(np.asarray(Xn[:n_tr][itr])).astype("float32")
    Xva = qt.transform(np.asarray(Xn[:n_tr][iva])).astype("float32")
    Xte = qt.transform(np.asarray(Xn[n_tr:])).astype("float32")
    Ctr = np.asarray(C[:n_tr][itr]); Cva = np.asarray(C[:n_tr][iva]); Cte = np.asarray(C[n_tr:])
    ytr, yva = y[itr].astype("float32"), y[iva]

    dev = get_device()
    T = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a).astype(dt)).to(dev)
    dtr, ntr_, ctr, ytr_t = T(Dtr, np.int64), T(Xtr, np.float32), T(Ctr, np.int64), T(ytr, np.float32)
    dva, nva, cva = T(Dva, np.int64), T(Xva, np.float32), T(Cva, np.int64)
    dte, nte, cte = T(Dte, np.int64), T(Xte, np.float32), T(Cte, np.int64)
    print(f"[f{fold_idx}] enc={args.enc} dev={dev} train={len(itr)} val={len(iva)} "
          f"nnum={Xtr.shape[1]} card={card} ({time.time()-t0:.0f}s)", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    variants = {"last": [np.zeros(len(iva)), np.zeros(len(Dte))],
                "argmax": [np.zeros(len(iva)), np.zeros(len(Dte))],
                f"top{args.keep}": [np.zeros(len(iva)), np.zeros(len(Dte))]}
    traces = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = CNN(Xtr.shape[1], Ctr.shape[1], card, dim=args.dim, emb=args.emb,
                p=args.dropout, arch=args.arch).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
        bs = args.bs
        steps = args.epochs * ((len(itr) + bs - 1) // bs)
        warm = max(1, int(0.05 * steps))
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else
            0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        lossf = make_loss(args.loss)
        g = torch.Generator().manual_seed(seed)
        top, trace, last_val = [], [], None
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(len(dtr), generator=g).to(dev)
            for i in range(0, len(dtr), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(m(dtr[idx], ntr_[idx], ctr[idx]), ytr_t[idx])
                loss.backward(); opt.step(); sch.step()
            pv = _predict(m, dva, nva, cva, dev)
            a = roc_auc_score(yva, pv)
            trace.append(float(a)); last_val = pv
            top.append((a, pv, copy.deepcopy({k: v.detach().cpu().clone()
                                              for k, v in m.state_dict().items()})))
            top = sorted(top, key=lambda t: -t[0])[: args.keep]
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"  f{fold_idx} s{seed} ep{ep+1}/{args.epochs} {a:.6f} "
                      f"best {top[0][0]:.6f} ({time.time()-t0:.0f}s)", flush=True)
        variants["last"][0] += last_val / len(seeds)
        m.load_state_dict({k: v.to(dev) for k, v in
                           copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()}).items()})
        variants["last"][1] += _predict(m, dte, nte, cte, dev) / len(seeds)
        b = max(top, key=lambda t: t[0])
        variants["argmax"][0] += b[1] / len(seeds)
        m.load_state_dict({k: v.to(dev) for k, v in b[2].items()})
        variants["argmax"][1] += _predict(m, dte, nte, cte, dev) / len(seeds)
        tk = f"top{args.keep}"
        variants[tk][0] += np.mean([t[1] for t in top], axis=0) / len(seeds)
        for t in top:
            m.load_state_dict({k: v.to(dev) for k, v in t[2].items()})
            variants[tk][1] += _predict(m, dte, nte, cte, dev) / (len(seeds) * len(top))
        traces[str(seed)] = {"per_epoch": [round(v, 6) for v in trace],
                             "best_auc": float(top[0][0]), "last_auc": float(trace[-1])}
        print(f"  f{fold_idx} s{seed} done best={top[0][0]:.6f} last={trace[-1]:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {k: float(roc_auc_score(yva, v[0])) for k, v in variants.items()}
    for k in variants:
        np.save(os.path.join(OUT, f"v5_{args.tag}_{k}_f{fold_idx}_oof.npy"), variants[k][0])
        np.save(os.path.join(OUT, f"v5_{args.tag}_{k}_f{fold_idx}_test.npy"), variants[k][1])
    np.save(os.path.join(OUT, f"v5_{args.tag}_f{fold_idx}_idx.npy"), iva)
    with open(os.path.join(OUT, f"v5_{args.tag}_f{fold_idx}.json"), "w") as f:
        json.dump({"fold": fold_idx, "enc": args.enc, "arch": args.arch,
                   "epochs": args.epochs,
                   "dim": args.dim, "emb": args.emb, "dropout": args.dropout,
                   "variant_auc": res, "best_auc": max(res.values()),
                   "traces": traces, "secs": round(time.time() - t0, 1)}, f)
    print(f"[{args.tag} f{fold_idx}] " + " ".join(f"{k}={v:.6f}" for k, v in res.items())
          + f" ({time.time()-t0:.0f}s)", flush=True)


def assemble(args):
    from sklearn.metrics import roc_auc_score
    meta = json.load(open(os.path.join(OUT, "v5_meta.json")))
    n_tr, n_te = meta["n_tr"], meta["n_te"]
    y = np.load(os.path.join(OUT, "v5_y.npy"))
    want = args.variant
    if want == "auto":
        tot = {}
        for f in range(5):
            p = os.path.join(OUT, f"v5_{args.tag}_f{f}.json")
            if os.path.exists(p):
                for k, v in json.load(open(p))["variant_auc"].items():
                    tot.setdefault(k, []).append(v)
        if not tot:
            print("no fold json yet"); return
        want = max(tot, key=lambda k: np.mean(tot[k]))
        print("mean fold AUC by variant:",
              {k: round(float(np.mean(v)), 6) for k, v in tot.items()}, "->", want, flush=True)
    oof = np.full(n_tr, np.nan); pred = np.zeros(n_te); got = 0
    for f in range(5):
        pf = os.path.join(OUT, f"v5_{args.tag}_{want}_f{f}_oof.npy")
        if not os.path.exists(pf):
            print(f"missing fold {f}"); continue
        oof[np.load(os.path.join(OUT, f"v5_{args.tag}_f{f}_idx.npy"))] = np.load(pf)
        pred += np.load(os.path.join(OUT, f"v5_{args.tag}_{want}_f{f}_test.npy")) / 5
        got += 1
    if got < 5 or np.isnan(oof).any():
        print(f"incomplete: {got}/5"); return
    assert len(oof) == 699960, len(oof)
    assert len(pred) == 299984, len(pred)
    assert np.isfinite(oof).all() and np.isfinite(pred).all()
    assert 0 < oof.min() and oof.max() < 1 and 0 < pred.min() and pred.max() < 1
    auc = roc_auc_score(y, oof)
    np.save(os.path.join(OUT, f"oof_{args.tag}.npy"), oof.astype("float64"))
    np.save(os.path.join(OUT, f"pred_{args.tag}.npy"), pred.astype("float64"))
    with open(os.path.join(OUT, f"score_{args.tag}.json"), "w") as f:
        json.dump({"tag": args.tag, "enc": args.enc, "arch": args.arch, "loss": args.loss, "epochs": args.epochs,
                   "seeds": args.seeds, "variant": want, "auc": float(auc),
                   "n_oof": int(len(oof)), "n_pred": int(len(pred)),
                   "oof_min": float(oof.min()), "oof_max": float(oof.max()),
                   "pred_min": float(pred.min()), "pred_max": float(pred.max())}, f, indent=1)
    print(f"[{args.tag}] OOF AUC = {auc:.6f} variant={want} written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "fold", "assemble"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--enc", default="raw", choices=["raw", "zeros", "norm"])
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,17,31415")
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--arch", default="deepsets", choices=["deepsets", "flat"])
    ap.add_argument("--loss", default="bce", choices=["bce","focal","pos"])
    ap.add_argument("--keep", type=int, default=3)
    args = ap.parse_args()
    if args.mode == "build":
        build_cache()
    elif args.mode == "fold":
        run_fold(args.fold, args)
    else:
        assemble(args)