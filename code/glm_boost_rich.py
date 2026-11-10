"""GLM-boost on a STRONGER design than last time.

Last attempt (gb_C1 0.950386) used only my 126-column digit-trie design, and a plain
LogisticRegression as the offset model. The public bank's gb* family reaches 0.9515 -
so either their design is richer, or their offset model and boosting schedule differ.

Two changes worth testing, both now affordable on 192 cores:

1. RICHER DESIGN: concatenate digit-trie TE (my 126 cols) WITH per-value TE at three
   priors (the axis-A winner, 42 cols) and the frequency features (axis B, 18 cols).
   Axis A alone scored 0.950518, above digit-trie's 0.950487, so the union should be
   strictly more informative.

2. BAGGED GLM-BOOST: the bank's gb* members are "bag5/bag10" - averages over bags.
   A single fit has variance; averaging 5 seeds/bags is exactly what took their family
   from ~0.950 to 0.9515. Do that.

Offset model: regularized LogisticRegression on the standardised design, per fold.
Boosting: LightGBM 4 leaves started from the GLM logit (init_score).
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


def build_rich_design():
    """digit-trie + per-value TE + frequency features, cached."""
    cf = os.path.join(OUT, "design_rich.npz")
    if os.path.exists(cf):
        z = np.load(cf)
        return z["X"], z["T"]
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr["is_fraud"].astype(int).values
    n_tr = len(tr)
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_tr), y))
    ymean = y.mean()
    BOOL = [c for c in tr.columns if tr[c].dtype == bool]
    NUM = [c for c in tr.columns if c not in CAT + BOOL + ["id", "is_fraud"]]
    full = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)
    full["n_flags"] = full[BOOL].sum(axis=1).astype("int8")
    NUM_ALL = NUM + ["n_flags"]

    def te_col(keys, prior):
        col = np.zeros(len(full))
        ks = pd.Series(keys)
        for itr, iva in folds:
            agg = pd.DataFrame({"v": y[itr], "k": ks.iloc[itr].values}).groupby("k")["v"].agg(["sum", "count"])
            sm = (agg["sum"] + prior * ymean) / (agg["count"] + prior)
            col[iva] = ks.iloc[iva].map(sm).fillna(ymean).values
        agg = pd.DataFrame({"v": y, "k": ks.iloc[:n_tr].values}).groupby("k")["v"].agg(["sum", "count"])
        sm = (agg["sum"] + prior * ymean) / (agg["count"] + prior)
        col[n_tr:] = ks.iloc[n_tr:].map(sm).fillna(ymean).values
        return col.astype("float32")

    def dkeys(s, i_d=4, d_d=3):
        v = np.abs(s.values.astype("float64"))
        ip = np.floor(v).astype("int64")
        fr = np.round((v - ip) * 10 ** d_d).astype("int64")
        o = {}
        for d in range(1, i_d + 1):
            o[f"i{d}"] = np.array([str(x)[:d] for x in ip])
        for d in range(1, d_d + 1):
            div = 10 ** (d_d - d)
            o[f"d{d}"] = np.array([f"{i}.{f // div:0{d}d}" for i, f in zip(ip, fr)])
        return o

    feats = {}
    # digit-trie
    for c in NUM_ALL:
        for depth, arr in dkeys(full[c]).items():
            feats[f"dt_{c[:7]}_{depth}"] = te_col(arr, 100.0)
    # per-value TE (the axis-A winner)
    for c in NUM_ALL:
        key = full[c].astype(str).values
        for prior in [30.0, 100.0, 300.0]:
            feats[f"pv_{c[:8]}_p{int(prior)}"] = te_col(key, prior)
    # frequency
    for c in NUM_ALL:
        key = pd.Series(full[c].astype(str).values)
        vc = key.value_counts()
        feats[f"fq_{c[:8]}"] = key.map(vc).values.astype("float32")

    F = pd.DataFrame(feats)
    for c in CAT:
        full[c] = pd.Categorical(full[c]).codes.astype("int32")
    for c in BOOL:
        full[c] = full[c].astype("int8")
    base = full.drop(columns=["id"])
    allX = pd.concat([base.reset_index(drop=True), F.reset_index(drop=True)], axis=1).astype("float32")
    X = allX.iloc[:n_tr].values
    T = allX.iloc[n_tr:].values
    np.savez_compressed(cf, X=X, T=T)
    with open(os.path.join(OUT, "design_rich_cols.json"), "w") as f:
        json.dump(list(allX.columns), f)
    return X, T


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    t0 = time.time()
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["is_fraud"]).is_fraud.values.astype(int)
    X, T = build_rich_design()
    X = X.astype("float64"); T = T.astype("float64")
    print(f"rich design {X.shape} built/loaded ({time.time()-t0:.0f}s)", flush=True)
    cols = json.load(open(os.path.join(OUT, "design_rich_cols.json")))
    cat_idx = [i for i, c in enumerate(cols) if c in CAT]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    NBAG = 5
    oof = np.zeros(len(y)); pred = np.zeros(len(T))
    for fi, (itr, iva) in enumerate(folds):
        sc = StandardScaler().fit(X[itr])
        Xtr = sc.transform(X[itr]); Xva = sc.transform(X[iva]); Xte = sc.transform(T)
        glm = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs", n_jobs=8)
        glm.fit(Xtr, y[itr])
        init_tr = glm.decision_function(Xtr)
        init_va = glm.decision_function(Xva)
        init_te = glm.decision_function(Xte)
        glm_auc = roc_auc_score(y[iva], init_va)
        va_acc = np.zeros(len(iva)); te_acc = np.zeros(len(T))
        for b in range(NBAG):
            P = dict(objective="binary", metric="auc", learning_rate=0.03,
                     num_leaves=4, min_child_samples=100, colsample_bytree=0.5,
                     subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                     num_threads=32, verbose=-1, seed=100 + b, force_col_wise=True)
            ds = lgb.Dataset(X[itr], y[itr], init_score=init_tr, categorical_feature=cat_idx)
            m = lgb.train(P, ds, num_boost_round=2500)
            va_acc += (init_va + m.predict(X[iva], raw_score=True)) / NBAG
            te_acc += (init_te + m.predict(T, raw_score=True)) / NBAG
        oof[iva] = 1.0 / (1.0 + np.exp(-va_acc))
        pred += (1.0 / (1.0 + np.exp(-te_acc))) / len(folds)
        print(f"  fold {fi}: GLM {glm_auc:.6f} -> gb {roc_auc_score(y[iva], oof[iva]):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    auc = roc_auc_score(y, oof)
    np.save(os.path.join(OUT, "oof_gb_rich.npy"), oof)
    np.save(os.path.join(OUT, "pred_gb_rich.npy"), pred)
    with open(os.path.join(OUT, "score_gb_rich.json"), "w") as f:
        json.dump({"tag": "gb_rich", "auc": float(auc), "nbag": NBAG,
                   "ncol": int(X.shape[1]), "secs": round(time.time()-t0, 1)}, f)
    print(f"\n[gb_rich] OOF AUC = {auc:.6f}")
    print("reference: bank best single 0.951496 | my digit-trie 0.950487 | "
          "prev gb attempt 0.950386")


if __name__ == "__main__":
    main()
