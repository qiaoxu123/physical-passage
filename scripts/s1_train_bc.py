"""S1 training: class-weighted cross-entropy on the expert dataset.

Validation split is by EPISODE (not frame) to avoid near-duplicate leakage.

    conda activate habvln   # torch + CUDA
    python scripts/s1_train_bc.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physical_passage.agents.bc_cnn import PassageCNN
from physical_passage.envs.actions import ACTIONS, N_ACTIONS

RESULTS = Path(__file__).resolve().parent.parent / "results"
WEIGHTS = RESULTS / "weights" / "bc_policy.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["/data/physical-passage/bc_data/train.npz"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-proprio", action="store_true",
                    help="image-only ablation (ignore stored states)")
    ap.add_argument("--aux-weight", type=float, default=2.0,
                    help="weight of the [dx,dz,rot_rem,feasible] regression loss")
    ap.add_argument("--out", default=str(WEIGHTS))
    args = ap.parse_args()

    ds = [np.load(p) for p in args.data]
    x = np.concatenate([d["images"] for d in ds])
    y = np.concatenate([d["labels"] for d in ds])
    ep = np.concatenate([d["episodes"] for d in ds])
    use_proprio = all("states" in d.files for d in ds) and not args.no_proprio
    s = (np.concatenate([d["states"] for d in ds]) if use_proprio
         else np.zeros((len(y), 0), dtype=np.float32))
    use_aux = all("aux" in d.files for d in ds) and args.aux_weight > 0
    aux = (np.concatenate([d["aux"] for d in ds]) if use_aux
           else np.zeros((len(y), 4), dtype=np.float32))
    val_mask = (ep % 10) == 0                      # ~10% of episodes held out
    print(f"{x.shape[0]} frames, train {np.sum(~val_mask)} / val {np.sum(val_mask)}"
          f", proprio={'on' if use_proprio else 'off'}"
          f", aux={'on' if use_aux else 'off'}")

    def to_ds(mask):
        xt = torch.from_numpy(x[mask]).permute(0, 3, 1, 2).float() / 255.0
        return TensorDataset(xt, torch.from_numpy(s[mask]),
                             torch.from_numpy(aux[mask]),
                             torch.from_numpy(y[mask]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(to_ds(~val_mask), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(to_ds(val_mask), batch_size=args.batch)

    counts = np.bincount(y[~val_mask], minlength=N_ACTIONS).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / N_ACTIONS, 0.0)
    w = np.clip(w, 0.25, 8.0)
    print("class weights:", {ACTIONS[i]: round(float(v), 2)
                             for i, v in enumerate(w) if counts[i] > 0})

    in_ch = x.shape[3]
    model = PassageCNN(state_dim=s.shape[1], in_ch=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32,
                                                    device=device))
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = n = 0.0
        for xb, sb, ab, yb in train_dl:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            logits, aux_pred = model(xb, sb if use_proprio else None,
                                     with_aux=True)
            loss = lossf(logits, yb)
            if use_aux:
                loss = loss + args.aux_weight * nn.functional.mse_loss(
                    aux_pred, ab.to(device))
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(yb), n + len(yb)
        sched.step()

        model.eval()
        correct = total = 0
        per = np.zeros((N_ACTIONS, 2), dtype=np.int64)     # [correct, total]
        with torch.no_grad():
            for xb, sb, _ab, yb in val_dl:
                pred = model(xb.to(device),
                             sb.to(device) if use_proprio else None).argmax(1).cpu()
                correct += int((pred == yb).sum())
                total += len(yb)
                for c in range(N_ACTIONS):
                    m = yb == c
                    per[c] += [int((pred[m] == c).sum()), int(m.sum())]
        print(f"epoch {epoch+1:2d}  loss {tot/n:.4f}  val_acc {correct/total:.3f}"
              f"  ({time.time()-t0:.0f}s)")

    print("\nper-class val accuracy:")
    for c in range(N_ACTIONS):
        if per[c, 1]:
            print(f"  {ACTIONS[c]:20s} {per[c,0]:5d}/{per[c,1]:5d}"
                  f"  {per[c,0]/per[c,1]:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "state_dim": s.shape[1],
                "in_ch": in_ch,
                "val_acc": correct / total,
                "frames": int(x.shape[0])}, out_path)
    print(f"\nsaved {out_path}  (val_acc {correct/total:.3f})")


if __name__ == "__main__":
    main()
