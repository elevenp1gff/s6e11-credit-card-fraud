"""S6E11 large-scale hyperparameter grid on the Ascend server (192 cores, aarch64).

Goal: find configurations that beat the local frontier (OOF 0.950487 for a single
digit-trie LGBM). The server has 6x the cores, so a grid that takes a full day locally
runs in a couple of hours there.

Design (identical to the winning local one, so results are comparable):
  digit-trie target encoding of every numeric column at integer depths 1-4 and
  decimal depths 1-3, plus 3 cross-column digit keys, prior=100.
  Split: StratifiedKFold(5, shuffle=True, random_state=42).

Parallelism plan (per the server manual):
  - one worker process per grid point, each limited to 1-2 BLAS threads
  - LightGBM num_threads set explicitly
  - every worker writes its own JSON summary so a crash is localised

Outputs: one .npy OOF + .npy test prediction + .json score per grid point,
plus a pooled results.json.
"""
import os, sys, json, time, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.environ.get("S6E11_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
CODE = os.path.join(ROOT, "code")

CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]

def build_design(cache=True):
    """Return X, T, y, folds with digit-trie features. Cached to npz."""
    cf = os.path.join(OUT, "design.npz")
    yf = os.path.join(OUT, "y.npy")
    if cache and os.path.exists(cf) and os.path.exists(yf):
        z = np.load(cf)
        return z["X"], z["T"], np.load(yf)

    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr["is_fraud"].astype(int).values
    n_tr = len(tr)
    BOOL = [c for c in tr.columns if tr[c].dtype == bool]
    NUM = [c for c in tr.columns if c not in CAT + BOOL + ["id", "is_fraud"]]

    full = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)
    full["n_flags"] = full[BOOL].sum(axis=1).astype("int8")
    NUM_ALL = NUM + ["n_flags"]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n_tr), y))
    ymean = y.mean()

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

    def te(keys, prior):
        col = np.zeros(len(full))
        for itr, iva in folds:
            agg = pd.DataFrame({"v": y[itr], "k": keys[itr]}).groupby("k")["v"].agg(["sum", "count"])
            sm = (agg["sum"] + prior * ymean) / (agg["count"] + prior)
            col[iva] = pd.Series(keys[iva]).map(sm).fillna(ymean).values
        agg = pd.DataFrame({"v": y, "k": keys[:n_tr]}).groupby("k")["v"].agg(["sum", "count"])
        sm = (agg["sum"] + prior * ymean) / (agg["count"] + prior)
        col[n_tr:] = pd.Series(keys[n_tr:]).map(sm).fillna(ymean).values
        return col.astype("float32")

    d = full.copy()
    feats = {}
    for c in NUM_ALL:
        for depth, arr in dkeys(full[c]).items():
            feats[f"{c[:7]}_{depth}"] = te(arr, 100.0)
    for c1, c2 in [("merchant_risk_score", "velocity_score"),
                   ("merchant_risk_score", "amount_usd"),
                   ("velocity_score", "amount_usd")]:
        k1 = dkeys(full[c1], 2, 1)["d1"]
        k2 = dkeys(full[c2], 2, 1)["d1"]
        cross = np.array([f"{a}|{b}" for a, b in zip(k1, k2)])
        feats[f"x_{c1[:6]}_{c2[:6]}"] = te(cross, 150.0)

    F = pd.DataFrame(feats)
    for c in CAT:
        d[c] = pd.Categorical(d[c]).codes.astype("int32")
    for c in BOOL:
        d[c] = d[c].astype("int8")

    base = d.drop(columns=["id"])
    allX = pd.concat([base.reset_index(drop=True), F.reset_index(drop=True)], axis=1).astype("float32")
    X = allX.iloc[:n_tr].values
    T = allX.iloc[n_tr:].values
    np.savez_compressed(cf, X=X, T=T)
    np.save(yf, y)
    with open(os.path.join(OUT, "design_cols.json"), "w") as f:
        json.dump(list(allX.columns), f)
    return X, T, y


GRID = []
for nl in [4, 6, 8]:
    for cs in [0.4, 0.5, 0.6]:
        for mcs in [50, 100, 200]:
            for lr in [0.02, 0.03]:
                GRID.append(dict(num_leaves=nl, colsample=cs, mcs=mcs, lr=lr))


def run_point(cfg, X, T, y, folds, cat_idx, nthread):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    P = dict(objective="binary", metric="auc", learning_rate=cfg["lr"],
             num_leaves=cfg["num_leaves"], min_child_samples=cfg["mcs"],
             colsample_bytree=cfg["colsample"], subsample=0.8, subsample_freq=1,
             reg_lambda=1.0, num_threads=nthread, verbose=-1, seed=42,
             force_col_wise=True)
    oof = np.zeros(len(X)); pred = np.zeros(len(T))
    for itr, iva in folds:
        ds = lgb.Dataset(X[itr], y[itr], categorical_feature=cat_idx)
        m = lgb.train(P, ds, num_boost_round=2500)
        oof[iva] = m.predict(X[iva])
        pred += m.predict(T) / len(folds)
    return roc_auc_score(y, oof), oof, pred


def main():
    idx = int(sys.argv[1])
    nthread = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cfg = GRID[idx]
    tag = f"nl{cfg['num_leaves']}_cs{int(cfg['colsample']*100)}_mcs{cfg['mcs']}_lr{int(cfg['lr']*100)}"
    done = os.path.join(OUT, f"score_{tag}.json")
    if os.path.exists(done):
        print(f"[skip] {tag}")
        return
    t0 = time.time()
    X, T, y = build_design()
    cols = json.load(open(os.path.join(OUT, "design_cols.json")))
    cat_idx = [i for i, c in enumerate(cols) if c in CAT]
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    auc, oof, pred = run_point(cfg, X, T, y, folds, cat_idx, nthread)
    np.save(os.path.join(OUT, f"oof_{tag}.npy"), oof)
    np.save(os.path.join(OUT, f"pred_{tag}.npy"), pred)
    with open(done, "w") as f:
        json.dump({**cfg, "tag": tag, "auc": float(auc), "secs": round(time.time() - t0, 1)}, f)
    print(f"[done] {tag} auc={auc:.6f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        t0 = time.time()
        X, T, y = build_design(cache=False)
        print(f"design {X.shape} {T.shape} in {time.time()-t0:.0f}s")
    elif sys.argv[1] == "list":
        for i, g in enumerate(GRID):
            print(i, g)
    else:
        main()
