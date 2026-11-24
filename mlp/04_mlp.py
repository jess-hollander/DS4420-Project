"""
Method 3: MLP classifier for P(home team wins).

Inputs: pre-game rolling differences (home minus away) in shooting (eFG%), turnovers,
rebounding, free throw rate, offensive/defensive rating, pace, recent win %, rest.
Home/away is built in: every row is from the home team's perspective, so the
network's bias term carries home-court advantage. Output: sigmoid win probability.

Baseline: always pick the home team.
Outputs: figs/mlp_confusion.png, figs/mlp_calibration.png, results/mlp_test.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score, brier_score_loss,
                             ConfusionMatrixDisplay)

TEST_SEASON = "2025-26"
Path("figs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

df = pd.read_csv("data/games_model.csv", parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")
FEATS = [c for c in df.columns if c.startswith("D_")]
train, test = df[df["SEASON"] < TEST_SEASON], df[df["SEASON"] == TEST_SEASON]
X_tr, y_tr, X_te, y_te = train[FEATS], train["HOME_WIN"], test[FEATS], test["HOME_WIN"]
print(f"Train {len(train):,} games | Test ({TEST_SEASON}) {len(test):,} games")

# Time-ordered CV so tuning never sees future games
mlp = GridSearchCV(
    make_pipeline(StandardScaler(),
                  MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)),
    {"mlpclassifier__hidden_layer_sizes": [(8,), (16,), (32,), (32, 16)],
     "mlpclassifier__alpha": [1e-4, 1e-2, 1e-1, 1.0]},     # alpha = L2 penalty
    cv=TimeSeriesSplit(n_splits=5), scoring="neg_log_loss", n_jobs=-1,
).fit(X_tr, y_tr)
print("Best params:", mlp.best_params_)

p = mlp.predict_proba(X_te)[:, 1]
home_rate_tr = y_tr.mean()
results = pd.DataFrame({
    "model": ["Always pick home team", "MLP"],
    "accuracy": [y_te.mean(), accuracy_score(y_te, (p >= 0.5).astype(int))],
    "log_loss": [log_loss(y_te, np.full(len(y_te), home_rate_tr)), log_loss(y_te, p)],
    "brier": [brier_score_loss(y_te, np.full(len(y_te), home_rate_tr)), brier_score_loss(y_te, p)],
    "auc": [0.5, roc_auc_score(y_te, p)],
})
results.to_csv("results/mlp_test.csv", index=False)
print("\n", results.round(3).to_string(index=False))
print(f"MLP picks the home team in {np.mean(p >= 0.5):.0%} of test games")
print(f"Test games: {len(test)} (target: 1,230)")

# Plot 1: confusion matrix
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_te, (p >= 0.5).astype(int),
                                        display_labels=["Away win", "Home win"],
                                        cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"MLP, {TEST_SEASON} test set")
fig.tight_layout()
fig.savefig("figs/mlp_confusion.png", dpi=200)

# Plot 2: calibration (do predicted probabilities match observed win rates?)
frac, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
ax.plot(mean_pred, frac, "o-", label="MLP")
ax.set_xlabel("Predicted home win probability")
ax.set_ylabel("Observed home win rate")
ax.set_title(f"MLP calibration, {TEST_SEASON}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/mlp_calibration.png", dpi=200)
