"""MLP control: same features, same folds, same schedule - but NO digit CNN.

Purpose is diagnostic, not a deliverable. v5's digit CNN plateaus at OOF 0.9453
while a LightGBM on the raw 19 columns reaches 0.9506. Two explanations remain:
  (a) the digit-CNN representation is the ceiling, or
  (b) the neural approach itself is the ceiling on this tabular problem.
If a plain MLP on the exact numeric values also lands ~0.945, then (b) holds and
no amount of digit-encoding work will close the gap. This decides that.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
try:
    import torch_npu
except Exception:
    pass

ROOT = os.environ.get("S6E11_ROOT",
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "out")
FIELDS = 13


class Out(nn.Module):
    def __init__(self, nin):
        super().__init__(); self.lin = nn.Linear(nin, 2)
    def forward(self, x): return self.lin(x)[:, 0]


class MLP(nn.Module):
    def __init__(self, n_num, n_cat, card, h=256, emb=16, p=0.15):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(card, emb) for _ in range(n_cat)])
        d = n_num + n_cat * emb
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(p), Out(h))
    def forward(self, xn, c):
        parts = [xn]
        for j, m in enumerate(self.cat_emb): parts.append(m(c[:, j]))
        return self.net(torch.cat(parts, 1)).squeeze(-1)


def main(fold, epochs=20, seed=42, dev="npu:0"):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import QuantileTransformer
    torch.set_num_threads(8); torch.manual_seed(seed)
    Xn = np.load(os.path.join(OUT, "v5_Xn.npy"), mmap_mode="r")
    C = np.load(os.path.join(OUT, "v5_C.npy"), mmap_mode="r")
    y = np.load(os.path.join(OUT, "v5_y.npy"))
    meta = json.load(open(os.path.join(OUT, "v5_meta.json")))
    n_tr, card = meta["n_tr"], meta["card"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_tr), y))
    itr, iva = folds[fold]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200000, random_state=0).fit(np.asarray(Xn[:n_tr][itr]))
    Xtr = qt.transform(np.asarray(Xn[:n_tr][itr])).astype("float32")
    Xva = qt.transform(np.asarray(Xn[:n_tr][iva])).astype("float32")
    Xte = qt.transform(np.asarray(Xn[n_tr:])).astype("float32")
    Ctr, Cva, Cte = np.asarray(C[:n_tr][itr]), np.asarray(C[:n_tr][iva]), np.asarray(C[n_tr:])
    T = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a).astype(dt)).to(dev)
    xtr, ctr, ytr = T(Xtr, np.float32), T(Ctr, np.int64), T(y[itr], np.float32)
    xva, cva = T(Xva, np.float32), T(Cva, np.int64)
    xte, cte = T(Xte, np.float32), T(Cte, np.int64)
    m = MLP(Xtr.shape[1], Ctr.shape[1], card).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    bs = 2048; steps = epochs * ((len(itr) + bs - 1) // bs); warm = max(1, int(0.05 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(seed)
    top = []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lf(m(xtr[idx], ctr[idx]), ytr[idx]).backward()
            opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(xva, cva)).float().cpu().numpy().astype("float64")
        a = roc_auc_score(y[iva], pv); top.append((a, pv))
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  mlp f{fold} ep{ep+1}/{epochs} {a:.6f}", flush=True)
    best = max(top, key=lambda t: t[0])
    avg = np.mean([t[1] for t in sorted(top, key=lambda t: -t[0])[:3]], axis=0)
    np.save(os.path.join(OUT, f"v5_mlp_f{fold}_oof.npy"), avg)
    np.save(os.path.join(OUT, f"v5_mlp_f{fold}_idx.npy"), iva)
    with open(os.path.join(OUT, f"v5_mlp_f{fold}.json"), "w") as f:
        json.dump({"fold": fold, "best_auc": float(best[0]),
                   "top3_auc": float(roc_auc_score(y[iva], avg)),
                   "per_epoch": [round(t[0], 6) for t in top]}, f)
    print(f"[mlp f{fold}] best={best[0]:.6f} top3={roc_auc_score(y[iva], avg):.6f}", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]), int(a[2]) if len(a) > 2 else 20,
         dev=os.environ.get("DEV", "npu:0"))