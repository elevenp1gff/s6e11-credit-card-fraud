"""CNN trained on a DIFFERENT split - the strongest diversity available to me.

Two independent axes of decorrelation at once:
  1. architecture: CNN over decimal digits (vs trees on target encodings)
  2. fold partition: seed 123 instead of 42

The public CNN helped because of axis 1. My split members helped (LB 0.95237 -> 0.95238)
because of axis 2. A model with both should be the most useful member I can build locally.

Use the v3 architecture (which reached 0.9449 on the standard split) but with
StratifiedKFold(seed=123). Its OOF is measured on seed-123 folds (comparable only within
itself) and its test prediction goes into the fixed-weight blend.

Run folds in parallel on the NPU cards that are free.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = os.path.join(os.environ.get("S6E11_ROOT",
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
OUT = os.path.join(os.environ.get("S6E11_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
N_INT, N_DEC = 5, 3
DIM = 48
SPLIT_SEED = 123


def get_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build():
    cf = os.path.join(OUT, "cnn3_design.npz")
    z = np.load(cf)
    return z["D"], z["N"], z["C"], z["y"], z["Dt"], z["Nt"], z["Ct"]


class CNN(nn.Module):
    def __init__(self, n_num, n_cat, card, emb=16, p=0.15):
        super().__init__()
        self.digit_emb = nn.Embedding(10, 8)
        self.conv = nn.Sequential(
            nn.Conv1d(8 * (N_INT + N_DEC), DIM, 3, padding=1), nn.SiLU(),
            nn.Conv1d(DIM, DIM, 3, padding=1), nn.SiLU())
        self.cat_emb = nn.ModuleList([nn.Embedding(card, emb) for _ in range(n_cat)])
        self.pool_proj = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        in_dim = DIM + n_num + n_cat * emb + DIM
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(p),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(p),
            nn.Linear(256, 1))

    def forward(self, d, nv, c):
        e = self.digit_emb(d)
        B, F, Dg, E = e.shape
        x = e.reshape(B, F, Dg * E).transpose(1, 2)
        h = self.conv(x)
        pooled = h.mean(dim=2)
        resid = self.pool_proj(h.max(dim=2).values)
        parts = [pooled, nv, resid]
        for j, m in enumerate(self.cat_emb):
            parts.append(m(c[:, j]))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


def main(fold_idx):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    D, Nv, C, y, Dt, Nt, Ct = build()
    card = int(max(C.max(), Ct.max())) + 1
    dev = get_device()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(np.zeros(len(y)), y))
    itr, iva = folds[fold_idx]
    print(f"device {dev} fold {fold_idx} split-seed {SPLIT_SEED} "
          f"train {len(itr)} val {len(iva)}", flush=True)

    SEEDS = [42, 17, 31415]
    av = np.zeros(len(iva)); at = np.zeros(len(Dt))
    T = lambda a, dt: torch.from_numpy(a.astype(dt)).to(dev)
    dtr, ntr, ctr, ytr = T(D[itr], np.int64), T(Nv[itr], np.float32), T(C[itr], np.int64), T(y[itr], np.float32)
    dva, nva, cva = T(D[iva], np.int64), T(Nv[iva], np.float32), T(C[iva], np.int64)
    dte, nte, cte = T(Dt, np.int64), T(Nt, np.float32), T(Ct, np.int64)

    for seed in SEEDS:
        torch.manual_seed(seed)
        m = CNN(D.shape[1], C.shape[1], card).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
        bs = 2048
        steps = 60 * ((len(itr) + bs - 1) // bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.3)
        lossf = nn.BCEWithLogitsLoss()
        best = (0, None, None); bad = 0
        for ep in range(60):
            m.train()
            perm = torch.randperm(len(dtr), device=dev)
            for i in range(0, len(dtr), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(m(dtr[idx], ntr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                opt.step()
                sch.step()
            m.eval()
            with torch.no_grad():
                pv = torch.sigmoid(m(dva, nva, cva)).cpu().numpy()
            a = roc_auc_score(y[iva], pv)
            if a > best[0]:
                with torch.no_grad():
                    pt = torch.sigmoid(m(dte, nte, cte)).cpu().numpy()
                best = (a, pv, pt); bad = 0
            else:
                bad += 1
            if ep % 10 == 0 or bad == 0:
                print(f"  s{seed} ep{ep+1} {a:.6f} best {best[0]:.6f}", flush=True)
            if bad >= 10:
                break
        av += best[1] / len(SEEDS); at += best[2] / len(SEEDS)
        print(f"seed {seed}: {best[0]:.6f} ({time.time()-t0:.0f}s)", flush=True)

    np.save(os.path.join(OUT, f"cnnS_f{fold_idx}_oof.npy"), av)
    np.save(os.path.join(OUT, f"cnnS_f{fold_idx}_test.npy"), at)
    np.save(os.path.join(OUT, f"cnnS_f{fold_idx}_idx.npy"), iva)
    with open(os.path.join(OUT, f"cnnS_f{fold_idx}.json"), "w") as f:
        json.dump({"fold": fold_idx, "split_seed": SPLIT_SEED,
                   "auc": float(roc_auc_score(y[iva], av)),
                   "secs": round(time.time()-t0, 1)}, f)
    print(f"[fold {fold_idx}] {roc_auc_score(y[iva], av):.6f}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
