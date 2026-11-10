"""Final deliverable: GBDT + decimal-CNN rank blend.

Weight choice: the OOF sweep is flat between w_cnn=0.15 and 0.225 (0.950912 /
0.950923 / 0.950923 / 0.950914), so the argmax is not meaningfully better than
anywhere on that plateau. Picking the argmax would be selection-optimistic on the
same OOF it is scored on, so w_cnn=0.15 is used: inside the plateau, on the
conservative side, and it does not depend on a precise OOF optimum.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("S6E11_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "out")
y = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["is_fraud"]).is_fraud.values.astype(int)
W = 0.15

# (rank - 0.5)/n maps the ranks strictly inside (0,1); rankdata/n would put the
# top row at exactly 1.0, which violates the required open interval.
r = lambda v: (rankdata(v) - 0.5) / len(v)
g_oof = r(np.load(os.path.join(OUT, "oof_gbr_e.npy")))
g_pr = r(np.load(os.path.join(OUT, "pred_gbr_e.npy")))
c_oof = r(np.load(os.path.join(OUT, "oof_v6cnn.npy")))
c_pr = r(np.load(os.path.join(OUT, "pred_v6cnn.npy")))

oof = (1 - W) * g_oof + W * c_oof
pred = (1 - W) * g_pr + W * c_pr
np.save(os.path.join(OUT, "oof_v6blend.npy"), oof.astype("float64"))
np.save(os.path.join(OUT, "pred_v6blend.npy"), pred.astype("float64"))
auc = roc_auc_score(y, oof)
with open(os.path.join(OUT, "score_v6blend.json"), "w") as f:
    json.dump({"tag": "v6blend",
               "components": {"gbr_e": 1 - W, "v6cnn": W},
               "detail": "rank-average; v6cnn = 6-seed decimal-digit CNN ensemble",
               "split": "StratifiedKFold(5, shuffle=True, random_state=42)",
               "auc": float(auc), "gbr_e_alone": float(roc_auc_score(y, g_oof))}, f, indent=1)
print(f"v6blend OOF AUC = {auc:.6f}  (gbr_e alone 0.950627)")

print("\n=== FINAL VERIFICATION of all deliverables (from saved .npy) ===")
for name in ["v6blend", "v6cnn", "raw", "gbr_e"]:
    o = np.load(os.path.join(OUT, f"oof_{name}.npy"))
    p = np.load(os.path.join(OUT, f"pred_{name}.npy"))
    ok = (len(o) == 699960 and len(p) == 299984 and np.isfinite(o).all()
          and np.isfinite(p).all() and o.min() > 0 and o.max() < 1
          and p.min() > 0 and p.max() < 1)
    print(f"{name:9s} auc={roc_auc_score(y, o):.6f} len={len(o)}/{len(p)} "
          f"range oof=({o.min():.2e},{o.max():.2e}) ALL_CHECKS={'PASS' if ok else 'FAIL'}")