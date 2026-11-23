"""
Trains and evaluates our Beta-Binomial model, then saves the results and figures.

Train/test split
----------------
    Training seasons : 2021-22, 2022-23, 2023-24, 2024-25
    Test season      : 2025-26
We only use the training seasons to pick the empirical Bayes prior and the
carry-over weight rho. Every prediction is made before the game happens using
only earlier games, and the test metrics only include 2025-26 games.

How to run it (from the main project folder):
    python data/fetch_nba_data.py        # only need to do this once, makes data/games.csv
    python -m beta_binomial.run
Everything gets saved to results/beta_binomial/.
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
CHECKPOINTS = [10, 20, 41]   # how many games into the season we check for the shrinkage analysis

# Colors and style settings for the plots
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def raw_winpct_probs(p):
    """
    Baseline that doesn't use a prior at all. It plugs each team's raw win %
    going into the game into the log5 formula, with the same home court term
    as the Bayesian model. A team's win % is set to 0.5 before its first game,
    and we clip it to [0.02, 0.98] so the log loss doesn't blow up.
    """
    def pct(w, l):
        n = w + l
        return np.clip(np.where(n > 0, w / np.maximum(n, 1), 0.5), 0.02, 0.98)
    z = (logit(pct(p["HOME_W"], p["HOME_L"])) - logit(pct(p["AWAY_W"], p["AWAY_L"]))
         + (1 - p["NEUTRAL"]) * logit(p["ETA_MEAN"]))
    return sigmoid(z).to_numpy()


def season_records_at(history, games_played):
    """Get each team's posterior and record at the point where it has played `games_played` games."""
    h = history.assign(GP=history["W"] + history["L"])
    h = h[h["GP"] == games_played]
    return h.sort_values("GAME_DATE").groupby(["SEASON", "TEAM"]).head(1)


def shrinkage_analysis(history, final_records, seasons):
    """
    Checks the main idea behind the project. After k games, which is better at
    predicting how a team does for the rest of the season: the posterior mean
    or their raw win %?
    """
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

    # Fit the prior on the training seasons
    rec_train = team_season_records(games[train_mask])
    prior = fit_prior_empirical_bayes(rec_train["wins"], rec_train["games"])
    print(f"\nEmpirical-Bayes prior: Beta({prior['alpha0']:.2f}, {prior['beta0']:.2f})  "
          f"mean={prior['mean']:.3f}, strength={prior['strength']:.1f} pseudo-games "
          f"(method of moments: {prior['mom_strength']:.1f})")

    # Tune rho on the training data.
    # The carry-over doesn't do anything in the first season since there's no
    # previous season, so we tune on 2022-23 through 2024-25.
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

    # Run the final model on all the data
    model = BetaBinomialNBA(prior["alpha0"], prior["beta0"], rho=best_rho, n_samples=2000, seed=42)
    preds, history = model.run(games)
    preds.to_csv(OUT / "predictions.csv", index=False)

    # Other versions of the model and baselines to compare against. They all
    # go through the same game-by-game process so the comparison is fair.
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

    # Accuracy at different points in the test season
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

    # Shrinkage analysis (to show that raw W-L records are noisy)
    final_rec = team_season_records(games)
    shrink = shrinkage_analysis(history, final_rec, games["SEASON"].unique()).round(4)
    shrink.to_csv(OUT / "shrinkage_analysis.csv", index=False)
    print("\nPredicting rest-of-season win % (all seasons):\n" + shrink.to_string(index=False))

    # Posteriors at the end of the test season
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

    make_figures(post, preds.loc[test], p_raw, history, prior, best_rho)
    print(f"\nAll outputs written to {OUT.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(post, test_preds, p_raw, history, prior, rho):
    # Figure 1: credible intervals at the end of the season compared to raw win %
    fig, ax = plt.subplots(figsize=(6.5, 7.2))
    y = np.arange(len(post))[::-1]
    ax.hlines(y, post["CI_LOW"], post["CI_HIGH"], color=BLUE, lw=2, label="95% credible interval")
    ax.plot(post["POST_MEAN"], y, "o", ms=6, color=BLUE, mec="white", mew=1.5, label="Posterior mean")
    ax.plot(post["RAW_WINPCT"], y, "D", ms=5, color=ORANGE, mec="white", mew=1, label="Raw win %")
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [f"{t} ({w:.0f}-{l:.0f})" for t, w, l in zip(post["TEAM"], post["W"], post["L"])],
                  fontsize=8)
    ax.set_xlabel("Team strength θ  (P(win vs. average team, neutral court))")
    ax.set_title(f"Posterior team strength, end of {TEST_SEASON}", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(OUT / "fig1_posterior_credible_intervals.png")
    plt.close(fig)

    # Figure 2: calibration on the test season
    min_n = 20  # skip bins with too few games to get a reliable win rate
    cal_bb = calibration_table(test_preds["HOME_WIN"], test_preds["P_HOME"], n_bins=10)
    cal_raw = calibration_table(test_preds["HOME_WIN"], p_raw, n_bins=10)
    cal_bb.to_csv(OUT / "calibration_beta_binomial.csv", index=False)
    cal_bb, cal_raw = cal_bb[cal_bb["n"] >= min_n], cal_raw[cal_raw["n"] >= min_n]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Perfect calibration")
    ax.plot(cal_raw["mean_pred"], cal_raw["obs_rate"], "-D", color=ORANGE, lw=2, ms=6,
            mec="white", label="Raw win % (log5)")
    ax.plot(cal_bb["mean_pred"], cal_bb["obs_rate"], "-o", color=BLUE, lw=2, ms=8,
            mec="white", label="Beta-Binomial")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home team wins)")
    ax.set_ylabel("Observed home win rate")
    ax.set_title(f"Calibration on {TEST_SEASON} test games", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.02, f"10 equal-width bins; bins with < {min_n} games omitted",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    fig.savefig(OUT / "fig2_calibration.png")
    plt.close(fig)

    # Figure 3 (extra): how the posterior changes over the season for 3 teams
    # (the best team, a middle team and the worst team)
    h = history[history["SEASON"] == TEST_SEASON]
    last = h.sort_values("GAME_DATE").groupby("TEAM").tail(1)
    last = last.assign(M=last["ALPHA"] / (last["ALPHA"] + last["BETA"])).sort_values("M")
    picks = [last["TEAM"].iloc[-1], last["TEAM"].iloc[len(last) // 2], last["TEAM"].iloc[0]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for team, col in zip(picks, [BLUE, AQUA, ORANGE]):
        d = h[h["TEAM"] == team].sort_values("GAME_DATE")
        d = d[d["W"] + d["L"] > 0].drop_duplicates(subset=["W", "L"])
        gp = d["W"] + d["L"]
        mean = d["ALPHA"] / (d["ALPHA"] + d["BETA"])
        sd = np.sqrt(d["ALPHA"] * d["BETA"] / ((d["ALPHA"] + d["BETA"]) ** 2 * (d["ALPHA"] + d["BETA"] + 1)))
        ax.fill_between(gp, mean - 2 * sd, mean + 2 * sd, color=col, alpha=0.15, lw=0)
        ax.plot(gp, mean, color=col, lw=2)
        ax.plot(gp, d["W"] / gp, color=col, lw=1, ls=":")
        ax.text(gp.iloc[-1] + 1, mean.iloc[-1], team, color=INK, va="center", fontsize=9)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.set_xlim(0, 90), ax.set_ylim(0, 1)
    ax.set_xlabel("Games played")
    ax.set_ylabel("Team strength θ")
    ax.set_title(f"Posterior mean ±2 SD (solid) vs raw win % (dotted), {TEST_SEASON}",
                 loc="left", fontsize=10, color=INK)
    fig.savefig(OUT / "fig3_posterior_evolution.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
