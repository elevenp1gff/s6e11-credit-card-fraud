"""Expand the GLM-boost family: the one newly-verified source of incremental signal.

Ablation evidence from the last run:
  bank + mine (with gb_rich)      CV 0.952268
  bank + mine (without gb_rich)   CV 0.952226
  -> gb_rich contributed +0.000042, i.e. it is NOT redundant with the 64-member bank.
That is remarkable given gb_rich's standalone AUC is only 0.950562. Its value is
structural: boosting from a GLM offset on a feature union no one else used.

So train a FAMILY of them, varying the axes most likely to decorrelate:
  - num_leaves 4 / 8 (bank used 3-4)
  - offset C 0.01 / 0.05 (offset strength changes the residual the trees must fit)
  - bags 5 / 10
  - boosting rounds 2500 / 4000
Each variant is saved separately so the stacker can weight them; too-similar variants
will simply get small weights.

Run in parallel: 8 variants, 24 threads each = 192 cores.
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

VARIANTS = [
    dict(nl=4, C=0.01, nbag=5, rounds=2500, tag="gbr_a"),
    dict(nl=8, C=0.01, nbag=5, rounds=2500, tag="gbr_b"),
    dict(nl=4, C=0.05, nbag=5, rounds=2500, tag="gbr_c"),
    dict(nl=4, C=0.01, nbag=10, rounds=2500, tag="gbr_d"),
    dict(nl=4, C=0.01, nbag=5, rounds=4000, tag="gbr_e"),
    dict(nl=3, C=0.01, nbag=5, rounds=3000, tag="gbr_f"),
    dict(nl=6, C=0.02, nbag=7, rounds=3000, tag="gbr_g"),
    dict(nl=4, C=0.005, nbag=8, rounds=2500, tag="gbr_h"),
]


def main(idx):
    cfg = VARIANTS[idx]
    tag = cfg["tag"]
    if os.path.exists(os.path.join(OUT, f"score_{tag}.json")):
        print(f"[skip] {tag}")
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    t0 = time.time()
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["is_fraud"]).is_fraud.values.astype(int)
    z = np.load(os.path.join(OUT, "design_rich.npz"))
    X = z["X"].astype("float64"); T = z["T"].astype("float64")
    cols = json.load(open(os.path.join(OUT, "design_rich_cols.json")))
    cat_idx = [i for i, c in enumerate(cols) if c in CAT]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    oof = np.zeros(len(y)); pred = np.zeros(len(T))
    for itr, iva in folds:
        sc = StandardScaler().fit(X[itr])
        Xtr = sc.transform(X[itr]); Xva = sc.transform(X[iva]); Xte = sc.transform(T)
        glm = LogisticRegression(C=cfg["C"], max_iter=500, solver="lbfgs", n_jobs=4)
        glm.fit(Xtr, y[itr])
        init_tr = glm.decision_function(Xtr)
        init_va = glm.decision_function(Xva)
        init_te = glm.decision_function(Xte)
        va_acc = np.zeros(len(iva)); te_acc = np.zeros(len(T))
        for b in range(cfg["nbag"]):
            P = dict(objective="binary", metric="auc", learning_rate=0.03,
                     num_leaves=cfg["nl"], min_child_samples=100, colsample_bytree=0.5,
                     subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                     num_threads=20, verbose=-1, seed=200 + b, force_col_wise=True)
            ds = lgb.Dataset(X[itr], y[itr], init_score=init_tr, categorical_feature=cat_idx)
            m = lgb.train(P, ds, num_boost_round=cfg["rounds"])
            va_acc += (init_va + m.predict(X[iva], raw_score=True)) / cfg["nbag"]
            te_acc += (init_te + m.predict(T, raw_score=True)) / cfg["nbag"]
        oof[iva] = 1.0 / (1.0 + np.exp(-va_acc))
        pred += (1.0 / (1.0 + np.exp(-te_acc))) / len(folds)

    auc = roc_auc_score(y, oof)
    np.save(os.path.join(OUT, f"oof_{tag}.npy"), oof)
    np.save(os.path.join(OUT, f"pred_{tag}.npy"), pred)
    with open(os.path.join(OUT, f"score_{tag}.json"), "w") as f:
        json.dump({**cfg, "auc": float(auc), "secs": round(time.time() - t0, 1)}, f)
    print(f"[{tag}] auc={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
