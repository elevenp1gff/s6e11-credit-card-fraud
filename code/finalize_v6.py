"""Finalise v6 deliverables: 6-seed decimal-CNN ensemble, plus verification.

Verification is run from the SAVED files only, so the reported AUC is exactly what
a consumer of these arrays would compute.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("S6E11_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "out")
y = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["is_fraud"]).is_fraud.values.astype(int)

a_oof = np.load(os.path.join(OUT, "_tmp_a_oof.npy")); a_pr = np.load(os.path.join(OUT, "_tmp_a_pr.npy"))
b_oof = np.load(os.path.join(OUT, "_tmp_b_oof.npy")); b_pr = np.load(os.path.join(OUT, "b_pr".rjust(0) and "_tmp_b_pr.npy"))

oof = 0.5 * a_oof + 0.5 * b_oof
pred = 0.5 * a_pr + 0.5 * b_pr

np.save(os.path.join(OUT, "oof_v6cnn.npy"), oof.astype("float64"))
np.save(os.path.join(OUT, "pred_v6cnn.npy"), pred.astype("float64"))
auc = roc_auc_score(y, oof)
with open(os.path.join(OUT, "score_v6cnn.json"), "w") as f:
    json.dump({"tag": "v6cnn", "model": "decimal-digit CNN, digit-axis conv + DeepSets",
               "enc": "raw decimal digits, PAD token, W_INT=5 W_DEC=2",
               "seeds": "42,17,31415,7,2024,999999", "epochs": "60 (bagA) / 24 (bagB)",
               "aggregation": "mean of top-3 epochs per seed, prob-average of 2 seed bags",
               "split": "StratifiedKFold(5, shuffle=True, random_state=42)",
               "auc": float(auc)}, f, indent=1)
print(f"written oof_v6cnn / pred_v6cnn  AUC={auc:.6f}")

# ---- independent verification from the saved arrays ----
print("\n=== VERIFICATION (from saved files) ===")
for name in ["v6cnn", "raw", "gbr_e"]:
    po, pp = os.path.join(OUT, f"oof_{name}.npy"), os.path.join(OUT, f"pred_{name}.npy")
    if not os.path.exists(po):
        print(f"{name}: missing"); continue
    o, p = np.load(po), np.load(pp)
    checks = {
        "len(oof)==699960": len(o) == 699960,
        "len(pred)==299984": len(p) == 299984,
        "no NaN/inf oof": bool(np.isfinite(o).all()),
        "no NaN/inf pred": bool(np.isfinite(p).all()),
        "oof in (0,1)": bool(o.min() > 0 and o.max() < 1),
        "pred in (0,1)": bool(p.min() > 0 and p.max() < 1),
        "dtype float64": o.dtype == np.float64 and p.dtype == np.float64,
    }
    print(f"\n{name}: AUC={roc_auc_score(y, o):.6f}")
    for k, v in checks.items():
        print(f"   {'OK ' if v else 'FAIL'} {k}")
    print(f"   all pass: {all(checks.values())}")