"""
Train and evaluate the Beta-Binomial model (train 2021-22 to 2024-25, test 2025-26).

Run from the project folder: python -m beta_binomial.run
Outputs are saved to results/beta_binomial/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.fetch_nba_data import load_games  # noqa: E402
from beta_binomial.model import (  # noqa: E402
    BetaBinomialNBA, beta_credible_interval, calibration_table, evaluate,
    fit_prior_empirical_bayes, logit, sigmoid, team_season_records,
)

OUT = ROOT / "results" / "beta_binomial"
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"
RHO_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CHECKPOINTS = [10, 20, 41]   # games played, for the shrinkage analysis

# Plot style
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def raw_winpct_probs(p):
    """Baseline: log5 on raw pre-game win % (clipped to [0.02, 0.98])."""
    def pct(w, l):
        n = w + l
        return np.clip(np.where(n > 0, w / np.maximum(n, 1), 0.5), 0.02, 0.98)
    z = (logit(pct(p["HOME_W"], p["HOME_L"])) - logit(pct(p["AWAY_W"], p["AWAY_L"]))
         + (1 - p["NEUTRAL"]) * logit(p["ETA_MEAN"]))
    return sigmoid(z).to_numpy()


def season_records_at(history, games_played):
    """Each team's posterior and record after `games_played` games."""
    h = history.assign(GP=history["W"] + history["L"])
    h = h[h["GP"] == games_played]
    return h.sort_values("GAME_DATE").groupby(["SEASON", "TEAM"]).head(1)


def shrinkage_analysis(history, final_records, seasons):
    """Compare posterior mean vs. raw win % at predicting rest-of-season win %."""
    rows = []
    fin = final_records.set_index(["SEASON", "TEAM"])
    for k in CHECKPOINTS:
        snap = season_records_at(history[history["SEASON"].isin(seasons)], k)
        snap = snap.join(fin, on=["SEASON", "TEAM"])
        rest_w = snap["wins"] - snap["W"]
        rest_n = snap["games"] - k
        target = rest_w / rest_n
        raw = snap["W"] / k
        post = snap["ALPHA"] / (snap["ALPHA"] + snap["BETA"])
        rows.append({
            "games_played": k, "n_team_seasons": len(snap),
            "raw_winpct_MAE": float(np.mean(np.abs(raw - target))),
            "posterior_mean_MAE": float(np.mean(np.abs(post - target))),
            "raw_winpct_RMSE": float(np.sqrt(np.mean((raw - target) ** 2))),
            "posterior_mean_RMSE": float(np.sqrt(np.mean((post - target) ** 2))),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games = load_games()
    train_mask = games["SEASON"].isin(TRAIN_SEASONS)
    print(f"{len(games):,} games | train {train_mask.sum():,} | test {(~train_mask).sum():,}")

    # Fit the empirical Bayes prior on training seasons
    rec_train = team_season_records(games[train_mask])
    prior = fit_prior_empirical_bayes(rec_train["wins"], rec_train["games"])
    print(f"\nEmpirical-Bayes prior: Beta({prior['alpha0']:.2f}, {prior['beta0']:.2f})  "
          f"mean={prior['mean']:.3f}, strength={prior['strength']:.1f} pseudo-games "
          f"(method of moments: {prior['mom_strength']:.1f})")

    # Tune rho on training seasons (skip 2021-22, which has no previous season)
    tune_seasons = TRAIN_SEASONS[1:]
    tune = []
    for rho in RHO_GRID:
        m = BetaBinomialNBA(prior["alpha0"], prior["beta0"], rho=rho, n_samples=0)
        p, _ = m.run(games[train_mask])
        sel = p["SEASON"].isin(tune_seasons)
        tune.append({"rho": rho, **evaluate(p.loc[sel, "HOME_WIN"], p.loc[sel, "P_HOME"])})
    tune = pd.DataFrame(tune)
    best_rho = float(tune.loc[tune["log_loss"].idxmin(), "rho"])
    tune.to_csv(OUT / "rho_tuning.csv", index=False)
    print(f"Best carry-over rho (train log loss): {best_rho:.2f}")

    # Run the final model
    model = BetaBinomialNBA(prior["alpha0"], prior["beta0"], rho=best_rho, n_samples=2000, seed=42)
    preds, history = model.run(games)
    preds.to_csv(OUT / "predictions.csv", index=False)

    # Reduced model versions for comparison
    variants = {
        "Beta-Binomial (EB prior + carry-over)": preds,
        "Beta-Binomial (EB prior, no carry-over)":
            BetaBinomialNBA(prior["alpha0"], prior["beta0"], rho=0.0, n_samples=2000, seed=42).run(games)[0],
        "Beta-Binomial (uniform Beta(1,1) prior)":
            BetaBinomialNBA(1.0, 1.0, rho=0.0, n_samples=2000, seed=42).run(games)[0],
    }

    test = preds["SEASON"] == TEST_SEASON
    y_test = preds.loc[test, "HOME_WIN"].to_numpy()
    train_home_rate = games.loc[train_mask, "HOME_WIN"].mean()

    results = {}
    for name, p in variants.items():
        results[name] = evaluate(y_test, p.loc[p["SEASON"] == TEST_SEASON, "P_HOME"])

    tp = preds.loc[test]
    p_raw = raw_winpct_probs(tp)
    results["Raw win % (log5)"] = evaluate(y_test, p_raw)
    results["Always pick home team"] = evaluate(y_test, np.full(len(y_test), train_home_rate))

    res_df = pd.DataFrame(results).T.round(4)
    res_df.index.name = f"model ({TEST_SEASON} test season, n={test.sum()})"
    res_df.to_csv(OUT / "test_metrics.csv")
    print("\n" + res_df.to_string())

    # Test accuracy by stage of season
    tp = tp.assign(GAME_NUM=np.maximum(tp["HOME_W"] + tp["HOME_L"], tp["AWAY_W"] + tp["AWAY_L"]) + 1,
                   P_RAW=p_raw)
    stage = pd.cut(tp["GAME_NUM"], [0, 20, 41, 61, 83], labels=["1-20", "21-41", "42-61", "62-82"])
    by_stage = tp.groupby(stage, observed=True).apply(
        lambda d: pd.Series({"n": len(d),
                             "beta_binomial_acc": evaluate(d["HOME_WIN"], d["P_HOME"])["accuracy"],
                             "raw_winpct_acc": evaluate(d["HOME_WIN"], d["P_RAW"])["accuracy"],
                             "home_team_acc": d["HOME_WIN"].mean()}),
        include_groups=False).round(3)
    by_stage.index.name = "team game number"
    by_stage.to_csv(OUT / "accuracy_by_stage.csv")
    print("\nTest accuracy by stage of season:\n" + by_stage.to_string())

    # Shrinkage analysis
    final_rec = team_season_records(games)
    shrink = shrinkage_analysis(history, final_rec, games["SEASON"].unique()).round(4)
    shrink.to_csv(OUT / "shrinkage_analysis.csv", index=False)
    print("\nPredicting rest-of-season win % (all seasons):\n" + shrink.to_string(index=False))

    # End-of-season posteriors
    last = history[history["SEASON"] == TEST_SEASON].sort_values("GAME_DATE").groupby("TEAM").tail(1)
    ci = beta_credible_interval(last["ALPHA"].to_numpy(), last["BETA"].to_numpy(), seed=1)
    post = last.assign(POST_MEAN=last["ALPHA"] / (last["ALPHA"] + last["BETA"]),
                       CI_LOW=ci[:, 0], CI_HIGH=ci[:, 1],
                       RAW_WINPCT=last["W"] / (last["W"] + last["L"]))
    post = post.sort_values("POST_MEAN", ascending=False)
    post[["TEAM", "W", "L", "RAW_WINPCT", "ALPHA", "BETA", "POST_MEAN", "CI_LOW", "CI_HIGH"]] \
        .round(4).to_csv(OUT / f"posteriors_{TEST_SEASON}.csv", index=False)

    eta_a, eta_b = model.final_eta_
    summary = {
        "prior": {k: float(v) for k, v in prior.items()},
        "best_rho": best_rho,
        "home_court_posterior": {"alpha": eta_a, "beta": eta_b, "mean": eta_a / (eta_a + eta_b)},
        "test_metrics": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nHome-court posterior: Beta({eta_a:.0f}, {eta_b:.0f}), mean {eta_a/(eta_a+eta_b):.3f}")

    # Calibration bins (table only)
    calibration_table(tp["HOME_WIN"], tp["P_HOME"]).to_csv(OUT / "calibration_beta_binomial.csv", index=False)

    make_figures(post, variants, p_raw, train_home_rate)
    print(f"\nAll outputs written to {OUT.relative_to(ROOT)}")


# Figures
def early_mask(p, max_game=20):
    """Games within both teams' first `max_game` games of the season."""
    gp = np.maximum(p["HOME_W"] + p["HOME_L"], p["AWAY_W"] + p["AWAY_L"]) + 1
    return (gp <= max_game).to_numpy()


def make_figures(post, variants, p_raw, home_rate):
    # Figure 1: test-season results table
    labels = {
        "Beta-Binomial (EB prior + carry-over)": "Beta-Binomial: EB prior + carry-over (final)",
        "Beta-Binomial (EB prior, no carry-over)": "Beta-Binomial: EB prior, no carry-over",
        "Beta-Binomial (uniform Beta(1,1) prior)": "Beta-Binomial: uniform Beta(1,1) prior",
    }
    rows = {}
    for name, p in variants.items():
        tp = p[p["SEASON"] == TEST_SEASON]
        y, e, pr = tp["HOME_WIN"].to_numpy(), early_mask(tp), tp["P_HOME"].to_numpy()
        rows[labels[name]] = {**evaluate(y, pr), "early": evaluate(y[e], pr[e])["accuracy"]}
    baselines = {"Baseline: raw win % (log5)": p_raw,
                 "Baseline: always pick home team": np.full(len(y), home_rate)}
    for name, pr in baselines.items():
        rows[name] = {**evaluate(y, pr), "early": evaluate(y[e], pr[e])["accuracy"]}

    df = pd.DataFrame(rows).T[["accuracy", "early", "log_loss", "brier"]]
    best = {"accuracy": df["accuracy"].max(), "early": df["early"].max(),
            "log_loss": df["log_loss"].min(), "brier": df["brier"].min()}
    cell = [[f"{v:.1%}" if k in ("accuracy", "early") else f"{v:.3f}" for k, v in r.items()]
            for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.axis("off")
    tbl = ax.table(cellText=cell, rowLabels=df.index, loc="center", cellLoc="center",
                   colLabels=["Accuracy", "Accuracy\n(games 1–20)", "Log loss", "Brier"],
                   rowLoc="left", edges="horizontal")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor("#c3c2b7")
        cl.set_linewidth(0.6)
        if r == 0:
            cl.set_text_props(weight="bold", color=INK)
            cl.set_height(cl.get_height() * 1.5)
        elif c >= 0 and np.isclose(df.iloc[r - 1, c], best[list(best)[c]]):
            cl.set_text_props(weight="bold")  # best value per column
    fig.savefig(OUT / "report_fig1_results_table.png")
    plt.close(fig)

    # Figure 2: end-of-season team strength (95% credible interval) vs. raw win %
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    y = np.arange(len(post))[::-1]
    ax.hlines(y, post["CI_LOW"], post["CI_HIGH"], color=BLUE, lw=2, label="95% credible interval")
    ax.plot(post["POST_MEAN"], y, "o", ms=5.5, color=BLUE, mec="white", mew=1.2, label="Posterior mean")
    ax.plot(post["RAW_WINPCT"], y, "D", ms=4.5, color=ORANGE, mec="white", mew=0.8, label="Raw win %")
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [f"{t} ({w:.0f}–{l:.0f})" for t, w, l in zip(post["TEAM"], post["W"], post["L"])],
                  fontsize=7.5)
    ax.set_ylim(-0.8, len(post) - 0.2)
    ax.set_xlabel("Team strength θ = P(beat an average team on a neutral court)", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(OUT / "report_fig2_team_strength.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
