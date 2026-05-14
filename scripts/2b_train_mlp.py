#!/usr/bin/env python3
"""
Phase 2B: Train the MLP Ambiguity Detector (Traffic Cop)
=========================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset

Objective:
    Train a PyTorch MLP to classify each query as Ambiguous (1) or Unambiguous (0)
    based on the 15 cross-encoder confidence scores from Phase 1C.

    Input:  15 sorted scores [s1, s2, ..., s15] from Baseline 2
    Output: 1 (Ambiguous → route to SQL execution) / 0 (Unambiguous → use CE Top-1)
    Loss:   BCELoss with Sigmoid output

Input:
    data/spider_mlp_training_data.csv    — Labeled data from Phase 2A

Output:
    models/spider_ambiguity_mlp/mlp.pt   — Trained PyTorch MLP weights
    models/spider_ambiguity_mlp/scaler.pkl — Feature scaler

Metrics Reported:
    Validation Accuracy, Precision, Recall, F1
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",   "spider_mlp_training_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "spider_ambiguity_mlp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM   = 15
HIDDEN_DIMS = [64, 32]
EPOCHS      = 100
LR          = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class AmbiguityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # Single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))   # Probability in [0, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    score_cols = [f"s{i+1}" for i in range(INPUT_DIM)]
    X = df[score_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    print(f"Loaded {len(df)} samples. Ambiguous: {y.sum():.0f} ({y.mean()*100:.1f}%)\n")

    # 2. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # 3. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Tensors
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    # 4. Model, loss, optimizer
    model     = AmbiguityMLP(INPUT_DIM, HIDDEN_DIMS)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 5. Train
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = (model(Xv) >= 0.5).float()
                acc   = (preds == yv).float().mean().item()
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mlp.pt"))
            print(f"  Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Acc: {acc*100:.2f}%")

    # 6. Final evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "mlp.pt")))
    model.eval()
    with torch.no_grad():
        probs = model(Xv).numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec  = recall_score(y_val, preds, zero_division=0)
    f1   = f1_score(y_val, preds, zero_division=0)

    print("\n" + "=" * 50)
    print("  PHASE 2B — MLP Training Complete")
    print("=" * 50)
    print(f"  Val Accuracy  : {acc*100:.2f}%")
    print(f"  Val Precision : {prec*100:.2f}%")
    print(f"  Val Recall    : {rec*100:.2f}%")
    print(f"  Val F1        : {f1*100:.2f}%")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
