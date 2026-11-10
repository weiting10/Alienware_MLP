#!/usr/bin/env python3
"""
Trains a 1D CNN to identify the neural network from its GPU power trace.

Steps:
  1. Dataset: combine the sweep folders into one CSV (or reuse an existing CSV).
  2. Split: shuffle, then divide into training / validation / test sets
     (stratified, so every class keeps the same share in each set).
  3. Train a 1D CNN on the power values; the validation set picks the best epoch.
  4. Evaluate on the test set: accuracy, per-class precision/recall, confusion matrix.

Usage:
    python train_classifier.py --folders released_out mlp_out bnn_out cnn_out
    python train_classifier.py --csv all_models.csv                 # reuse a built CSV
    python train_classifier.py --csv all_models.csv --label family  # other label columns
    python train_classifier.py --csv all_models.csv --split by-model

Outputs (in --out, default results_<label>_<split>/):
    confusion_matrix.png       counts and row-normalised percentages on the test set
    training_curves.png        loss and accuracy per epoch
    report.txt                 accuracy, per-class precision / recall / F1, split sizes
    test_predictions.csv       every test trace with its true and predicted label
    history.csv, model.pt      per-epoch numbers and the trained weights
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =========================================================================== data

def load_csv(path, label_col):
    """Returns power matrix X (traces x points), labels y (strings), and metadata rows."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        p_cols = [i for i, h in enumerate(header) if h.startswith("p") and h[1:].isdigit()]
        if label_col not in header:
            sys.exit(f"label column '{label_col}' not in {path}; available: "
                     f"{[h for h in header if not (h.startswith('p') and h[1:].isdigit())]}")
        li = header.index(label_col)
        meta_cols = [i for i in range(len(header)) if i not in p_cols]
        X, y, meta = [], [], []
        for row in reader:
            X.append([float(row[i]) if row[i] not in ("", "nan") else np.nan for i in p_cols])
            y.append(row[li])
            meta.append({header[i]: row[i] for i in meta_cols})
    return np.array(X, dtype=np.float32), y, meta


def preprocess(X, n_idle):
    """Per trace: fill missing values, then subtract the idle level (median of the first
    n_idle points, the recording before the pass starts) so traces start near 0 W."""
    X = X.copy()
    for i in range(len(X)):
        row = X[i]
        if np.isnan(row).all():
            row[:] = 0.0
            continue
        fill = np.nanmedian(row)
        row[np.isnan(row)] = fill
        X[i] = row - np.median(row[:max(1, n_idle)])
    return X


def split_random(y, fracs, seed):
    """Shuffle, then split each class separately by the given fractions (stratified)."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for c in sorted(set(y)):
        idx = [i for i, v in enumerate(y) if v == c]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * fracs[2])) if len(idx) >= 3 else 0
        n_val = max(1, round(len(idx) * fracs[1])) if len(idx) >= 3 else 0
        test += idx[:n_test]
        val += idx[n_test:n_test + n_val]
        train += idx[n_test + n_val:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def split_by_model(y, models, fracs, seed):
    """Whole networks go into one set: all traces of e.g. 'gpt2-medium' land in the same set,
    so the test set only contains networks the classifier never saw."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for c in sorted(set(y)):
        groups = sorted({models[i] for i, v in enumerate(y) if v == c})
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * fracs[2])) if len(groups) >= 3 else 0
        n_val = max(1, round(len(groups) * fracs[1])) if len(groups) >= 3 else 0
        test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
        for i, v in enumerate(y):
            if v != c:
                continue
            (test if models[i] in test_g else val if models[i] in val_g else train).append(i)
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


# =========================================================================== model

class TraceCNN(nn.Module):
    """1D CNN: three conv blocks (conv -> batch norm -> ReLU -> max pool), global average
    pooling over time, then a small fully connected classifier."""

    def __init__(self, n_classes, channels=(32, 64, 128), kernel=7, dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2), nn.BatchNorm1d(c),
                       nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c_in, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):                 # x: (batch, points)
        return self.head(self.pool(self.features(x.unsqueeze(1))))


def run_epoch(model, X, y, loss_fn, device, batch, opt=None):
    """One pass over (X, y). Trains if opt is given, otherwise only evaluates."""
    model.train(opt is not None)
    order = torch.randperm(len(X)) if opt is not None else torch.arange(len(X))
    total_loss, correct, preds = 0.0, 0, []
    with torch.set_grad_enabled(opt is not None):
        for k in range(0, len(X), batch):
            idx = order[k:k + batch]
            xb, yb = X[idx].to(device), y[idx].to(device)
            if opt is not None and len(idx) == 1:
                continue                   # batch norm can't train on a single trace
            out = model(xb)
            loss = loss_fn(out, yb)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(idx)
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            preds.append(p.cpu())
    return total_loss / len(X), correct / len(X), torch.cat(preds) if preds else torch.tensor([])


# =========================================================================== reporting

def confusion(y_true, y_pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def plot_confusion(m, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = m.sum(1, keepdims=True)
    pct = np.divide(m, rows, out=np.zeros_like(m, dtype=float), where=rows > 0)
    size = max(5, 0.6 * len(classes) + 3)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(pct * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45 if len(classes) > 4 else 0,
                       ha="right" if len(classes) > 4 else "center")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if len(classes) <= 25:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if m[i, j]:
                    dark = pct[i, j] > 0.55
                    ax.text(j, i, f"{m[i, j]}\n{pct[i, j] * 100:.0f}%", ha="center", va="center",
                            fontsize=9, color="white" if dark else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of true class")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in hist], color="#2a78d6", label="training")
    a1.plot(ep, [h["val_loss"] for h in hist], color="#eb6834", label="validation")
    a1.set_title("Loss", loc="left")
    a1.set_xlabel("Epoch")
    a2.plot(ep, [h["train_acc"] * 100 for h in hist], color="#2a78d6", label="training")
    a2.plot(ep, [h["val_acc"] * 100 for h in hist], color="#eb6834", label="validation")
    a2.set_title("Accuracy (%)", loc="left")
    a2.set_xlabel("Epoch")
    for a in (a1, a2):
        a.legend(frameon=False)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color="#e8e7e2")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =========================================================================== main

def main():
    p = argparse.ArgumentParser(description="Train a 1D CNN on GPU power traces.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--folders", nargs="+", help="sweep output folders to combine into one CSV first")
    src.add_argument("--csv", help="an existing combined CSV from build_dataset.py")
    p.add_argument("--dataset-out", default="all_models.csv", help="CSV written when using --folders")
    p.add_argument("--label", default="model_type",
                   help="column to predict: model_type (transformer/mlp/bnn/cnn), family, model, ...")
    p.add_argument("--split", choices=["random", "by-model"], default="random",
                   help="random = shuffle all traces then split; by-model = whole networks per set")
    p.add_argument("--fractions", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12, help="stop if validation loss doesn't improve")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-class-weights", action="store_true",
                   help="don't weight the loss by class size (by default rare classes count more)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="results folder (default: results_<label>_<split>)")
    args = p.parse_args()
    if abs(sum(args.fractions) - 1) > 1e-6:
        sys.exit("--fractions must add up to 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out or f"results_{args.label}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    # 1. dataset
    if args.folders:
        from build_dataset import build
        build(args.folders, args.dataset_out)
        csv_path = args.dataset_out
    else:
        csv_path = args.csv
    X, labels, meta = load_csv(csv_path, args.label)
    pre_s, grid_ms = float(meta[0]["pre_s"]), float(meta[0]["grid_ms"])
    X = preprocess(X, int(round(pre_s * 1000 / grid_ms)))
    classes = sorted(set(labels))
    if len(classes) < 2:
        sys.exit(f"only one class ({classes}) in '{args.label}'; need at least two to train")
    y = np.array([classes.index(v) for v in labels])
    print(f"\n{len(X)} traces x {X.shape[1]} points, {len(classes)} classes in '{args.label}':")
    for c in classes:
        print(f"  {c:30s} {labels.count(c)}")

    # 2. shuffle + split
    if args.split == "random":
        tr, va, te = split_random(labels, args.fractions, args.seed)
    else:
        tr, va, te = split_by_model(labels, [m["model"] for m in meta], args.fractions, args.seed)
    print(f"split ({args.split}): train {len(tr)}, validation {len(va)}, test {len(te)}")
    if not va or not te:
        sys.exit("validation or test set is empty: too few traces (or networks) per class")

    # scale with training-set statistics only, so nothing leaks from validation/test
    scale = float(X[tr].std()) or 1.0
    T = lambda idx: (torch.tensor(X[idx] / scale), torch.tensor(y[idx]))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = T(tr), T(va), T(te)

    # 3. train
    device = torch.device(args.device)
    model = TraceCNN(len(classes)).to(device)
    counts = np.bincount(y[tr], minlength=len(classes)).astype(float)
    weights = None if args.no_class_weights else torch.tensor(
        counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 3)

    best, best_state, hist, wait = float("inf"), None, [], 0
    print(f"\ntraining on {device} ...")
    for ep in range(1, args.epochs + 1):
        tl, ta, _ = run_epoch(model, Xtr, ytr, loss_fn, device, args.batch, opt)
        vl, vacc, _ = run_epoch(model, Xva, yva, loss_fn, device, args.batch)
        sched.step(vl)
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=vacc))
        mark = ""
        if vl < best - 1e-4:
            best, wait, mark = vl, 0, "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"  epoch {ep:3d}: train loss {tl:.3f} acc {ta * 100:5.1f}%  |  "
              f"val loss {vl:.3f} acc {vacc * 100:5.1f}%{mark}", flush=True)
        if wait >= args.patience:
            print(f"  stopping: no validation improvement for {args.patience} epochs")
            break
    model.load_state_dict(best_state)

    # 4. test
    _, test_acc, pred = run_epoch(model, Xte, yte, loss_fn, device, args.batch)
    pred = pred.numpy()
    m = confusion(yte.numpy(), pred, len(classes))
    lines = [f"label: {args.label}   split: {args.split}   dataset: {csv_path}",
             f"train {len(tr)} / validation {len(va)} / test {len(te)} traces",
             f"best epoch {min(hist, key=lambda h: h['val_loss'])['epoch']} of {len(hist)}",
             f"\nTEST ACCURACY: {test_acc * 100:.1f}%",
             f"balanced accuracy (mean recall over classes): "
             f"{np.mean([m[i, i] / m[i].sum() for i in range(len(classes)) if m[i].sum()]) * 100:.1f}%\n",
             f"{'class':30s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'test traces':>12s}"]
    for i, c in enumerate(classes):
        tp, col, row = m[i, i], m[:, i].sum(), m[i].sum()
        prec = tp / col if col else 0.0
        rec = tp / row if row else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lines.append(f"{c:30s} {prec:9.2f} {rec:7.2f} {f1:6.2f} {row:12d}")
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")

    plot_confusion(m, classes, out / "confusion_matrix.png",
                   f"Test set confusion matrix\n{args.label}, {args.split} split  |  "
                   f"accuracy {test_acc * 100:.1f}%  |  {len(te)} test traces")
    plot_curves(hist, out / "training_curves.png")
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    with open(out / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "model", "true", "predicted", "correct"])
        for k, i in enumerate(te):
            w.writerow([meta[i].get("trace_id", ""), meta[i].get("model", ""), classes[y[i]],
                        classes[pred[k]], y[i] == pred[k]])
    torch.save({"state_dict": model.state_dict(), "classes": classes, "scale": scale,
                "label": args.label, "n_points": X.shape[1]}, out / "model.pt")
    print(f"\nsaved confusion_matrix.png, training_curves.png, report.txt, test_predictions.csv, "
          f"history.csv and model.pt in {out}/")


if __name__ == "__main__":
    main()