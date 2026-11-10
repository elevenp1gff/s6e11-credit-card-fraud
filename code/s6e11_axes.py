"""S6E11: a genuinely different feature axis on the server's 192 cores.

The lesson from the last two days: adding models on the SAME feature design yields
almost nothing (+0.00008 over 22 variants), while a member built on a DIFFERENT
mechanism (GPT-2) gave +0.0011. So the value of the server is not more grid points -
it is the ability to test NEW feature axes that were computationally out of reach.

Axes worth testing, cheap on 192 cores but slow on a laptop:
  A. Per-value (not per-digit) target encoding with many priors, for every numeric
     column - the column has <=50k distinct values so this is feasible here.
  B. Out-of-fold frequency/count features at several granularities.
  C. Neighbourhood statistics: for each of the top k numerics, the OOF target rate of
     rows whose value lies within a sliding window (a smoothed kernel TE), which
     captures local structure between grid points.
  D. Rank-within-group features for every (categorical, numeric) pair.
  E. Row-wise aggregates over the digit-TE block (mean/max/min/std) so the model sees
     "how unusual is this row across all grid columns".

Each axis is evaluated with the SAME probe model and folds so deltas are directly
comparable to the registered 0.950487 feature-design baseline.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.environ.get("S6E11_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]


def load():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr["is_fraud"].astype(int).values
    return tr, te, y


def folds_of(n, y):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))


def te_encode(keys, y, folds, prior, ymean):
    n_tr = len(y)
    col = np.zeros(len(keys))
    kser = pd.Series(keys)
    for itr, iva in folds:
        agg = pd.DataFrame({"v": y[itr], "k": kser.iloc[itr].values}).groupby("k")["v"].agg(["sum", "count"])
        sm = (agg["sum"] + prior * ymean) / (agg["count"] + prior)
        col[iva] = kser.iloc[iva].map(sm).fillna(ymean).values
    agg = pd.DataFrame({"v": y, "k": kser.iloc[:n_tr].values}).groupby("k")["v"].agg(["sum", "count"])
    sm = (agg["sum"] + prior * ymean) / (agg["count"] + prior)
    col[n_tr:] = kser.iloc[n_tr:].map(sm).fillna(ymean).values
    return col.astype("float32")


def build_axis(axis):
    tr, te, y = load()
    n_tr = len(tr)
    ymean = y.mean()
    folds = folds_of(n_tr, y)
    BOOL = [c for c in tr.columns if tr[c].dtype == bool]
    NUM = [c for c in tr.columns if c not in CAT + BOOL + ["id", "is_fraud"]]
    full = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)
    full["n_flags"] = full[BOOL].sum(axis=1).astype("int8")
    NUM_ALL = NUM + ["n_flags"]
    feats = {}

    if axis == "A_pervalue":
        for c in NUM_ALL:
            key = full[c].astype(str).values
            for prior in [30.0, 100.0, 300.0]:
                feats[f"pv_{c[:8]}_p{int(prior)}"] = te_encode(key, y, folds, prior, ymean)

    elif axis == "B_freq":
        for c in NUM_ALL:
            key = pd.Series(full[c].astype(str).values)
            vc = key.value_counts()
            feats[f"fq_{c[:8]}"] = key.map(vc).values.astype("float32")
        for c in CAT:
            key = pd.Series(full[c].astype(str).values)
            vc = key.value_counts()
            feats[f"fq_{c[:8]}"] = key.map(vc).values.astype("float32")

    elif axis == "C_kernel":
        # sliding-window smoothed target rate on the strongest numerics
        TOP = ["merchant_risk_score", "velocity_score", "amount_usd",
               "account_balance_usd", "distance_from_home_km", "customer_age"]
        for c in TOP:
            v = full[c].values.astype("float64")
            order = np.argsort(v, kind="mergesort")
            vs = v[order]
            for frac in [0.002, 0.01]:
                bw = max(1, int(len(v) * frac))
                col = np.zeros(len(v))
                for itr, iva in folds:
                    ys = np.zeros(len(v)); mask = np.zeros(len(v), dtype=bool)
                    ys[itr] = y[itr]; mask[itr] = True
                    ysum = np.concatenate([[0.0], np.cumsum(ys[order])])
                    msum = np.concatenate([[0], np.cumsum(mask[order].astype(np.int64))])
                    lo = np.clip(np.arange(len(v)) - bw, 0, len(v))
                    hi = np.clip(np.arange(len(v)) + bw + 1, 0, len(v))
                    num = ysum[hi] - ysum[lo]
                    den = msum[hi] - msum[lo]
                    # map back: sorted positions -> original index
                    smooth = np.zeros(len(v))
                    smooth[order] = (num + 50 * ymean) / (den + 50)
                    col[iva] = smooth[iva]
                # for test rows use full-data smoothing
                ysum = np.concatenate([[0.0], np.cumsum(y[order])])
                lo = np.clip(np.arange(len(v)) - bw, 0, len(v))
                hi = np.clip(np.arange(len(v)) + bw + 1, 0, len(v))
                num = ysum[hi] - ysum[lo]
                full_smooth = np.zeros(len(v))
                full_smooth[order] = (num + 50 * ymean) / ((hi - lo) + 50)
                col[n_tr:] = full_smooth[n_tr:]
                feats[f"kern_{c[:8]}_f{int(frac*1000)}"] = col.astype("float32")

    elif axis == "D_rankgroup":
        for c in NUM_ALL[:8]:
            for g in ["merchant_category", "auth_method"]:
                r = full.groupby(g)[c].rank(pct=True)
                feats[f"rg_{c[:6]}_{g[:4]}"] = r.astype("float32").values

    return pd.DataFrame(feats), full, y, folds, n_tr, CAT, BOOL


def probe(Xa, Xb, y, folds, n_tr, tag):
    """Train the fixed probe on base design + extra columns."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    cf = os.path.join(OUT, "design.npz")
    z = np.load(cf)
    X, T = z["X"], z["T"]
    if Xa is not None:
        X = np.hstack([X, Xa.values.astype("float32")])
        T = np.hstack([T, Xb.values.astype("float32")])
    cols = json.load(open(os.path.join(OUT, "design_cols.json")))
    cat_idx = [i for i, c in enumerate(cols) if c in CAT]
    P = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=4,
             min_child_samples=100, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
             reg_lambda=1.0, num_threads=16, verbose=-1, seed=42, force_col_wise=True)
    oof = np.zeros(len(X)); pred = np.zeros(len(T))
    for itr, iva in folds:
        m = lgb.train(P, lgb.Dataset(X[itr], y[itr], categorical_feature=cat_idx),
                      num_boost_round=2500)
        oof[iva] = m.predict(X[iva])
        pred += m.predict(T) / len(folds)
    auc = roc_auc_score(y, oof)
    np.save(os.path.join(OUT, f"oof_axis_{tag}.npy"), oof)
    np.save(os.path.join(OUT, f"pred_axis_{tag}.npy"), pred)
    with open(os.path.join(OUT, f"score_axis_{tag}.json"), "w") as f:
        json.dump({"axis": tag, "auc": float(auc), "ncol": int(X.shape[1])}, f)
    print(f"[axis {tag}] auc={auc:.6f} ncol={X.shape[1]}", flush=True)
    return auc


if __name__ == "__main__":
    axis = sys.argv[1]
    t0 = time.time()
    F, full, y, folds, n_tr, _, _ = build_axis(axis)
    print(f"built {axis}: {F.shape} in {time.time()-t0:.0f}s", flush=True)
    probe(F.iloc[:n_tr], F.iloc[n_tr:], y, folds, n_tr, axis)
