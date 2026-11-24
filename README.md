# DS 4420 Project: Predicting NBA Game Outcomes

Jessica Hollander, Suhana Singh Madia, Rayd Hussain

A team's win-loss record doesn't always reflect how good it is, especially
early in the season. We estimate team strength and predict game winners with
three methods:

| # | Method | Implementation | Owner | Status |
|---|--------|----------------|-------|--------|
| 1 | Beta-Binomial (Bayesian) | manual (NumPy) | Jessica | done |
| 2 | AR(p) on team scoring | statsmodels | Suhana | done |
| 3 | MLP win/loss classifier | scikit-learn | Rayd | done |

## Setup

```bash
pip install -r requirements.txt
python data/fetch_nba_data.py      # download and clean the data -> data/games.csv
python -m beta_binomial.run        # run method 1 -> results/beta_binomial/
```

## Data (`data/`)

Data comes from [`nba_api`](https://github.com/swar/nba_api), which pulls from
NBA.com's official stats. We used the `LeagueGameLog` endpoint for the
2021-22 through 2025-26 regular seasons.

* `raw_team_game_logs.csv`: 12,300 rows, one per team per game, with box score stats.
* `games.csv`: 6,150 rows, one per game, with `GAME_ID, SEASON, GAME_DATE,
  NEUTRAL`, both teams' IDs, abbreviations and box scores (`HOME_*`, `AWAY_*`),
  `HOME_WIN` and `POINT_DIFF`. All three methods use this file.

**Cleaning:** `MATCHUP` marks the home team with "vs." and the away team with
"@". We joined each game's two rows on `GAME_ID` and checked that every game
had exactly one winner. The 10 neutral-site games (international games and NBA
Cup semifinals) list both teams with "@"; we kept them, set `NEUTRAL = 1`, and
labeled the alphabetically first team as "home".

## Method 1: Beta-Binomial (`beta_binomial/`)

Written by hand in NumPy, without scikit-learn, statsmodels, scipy.stats or
PyMC. The docstring at the top of `beta_binomial/model.py` summarizes the math.

* **Team strength:** θᵢ, the probability team i beats an average team on a
  neutral court, has a Beta(α, β) prior. Treating each game as a Bernoulli
  trial, the posterior after W wins and L losses is Beta(α + W, β + L).
* **Prior (empirical Bayes):** Beta(α₀, β₀) maximizes the Beta-Binomial
  marginal likelihood of the training team-season records, found with our own
  grid search.
* **Season carry-over** (beyond course content): each new season's prior is
  α₀ + ρ·W_prev, β₀ + ρ·L_prev, with ρ tuned on the training seasons.
* **Home court:** η has its own Beta prior, updated with every non-neutral game.
* **Prediction:** logit P(home win) = logit θ_home − logit θ_away + logit η,
  averaged over 2,000 posterior samples. Games are predicted one day at a time
  using only earlier games.
* **Train/test split:** train on 2021-22 to 2024-25 (to fit the prior and ρ),
  test on 2025-26.

### Outputs (`results/beta_binomial/`)

| File | Contents |
|------|----------|
| `test_metrics.csv` | 2025-26 accuracy, log loss and Brier score for the model, reduced versions and baselines |
| `accuracy_by_stage.csv` | test accuracy by stage of season |
| `shrinkage_analysis.csv` | posterior mean vs. raw win % at predicting rest-of-season win % |
| `rho_tuning.csv` | training log loss for each ρ tried |
| `posteriors_2025-26.csv` | each team's end-of-season posterior and 95% credible interval |
| `calibration_beta_binomial.csv` | test-season calibration bins |
| `predictions.csv` | pre-game posterior and predicted P(home win) for every game |
| `summary.json` | prior, ρ, home-court posterior and test metrics |
| `report_fig1_results_table.png` | report Figure 1: test results vs. baselines |
| `report_fig2_team_strength.png` | report Figure 2: end-of-season posteriors and 95% credible intervals vs. raw win % |

## Method 3: MLP win/loss classifier (`mlp/`)

Built with scikit-learn's `MLPClassifier`. Predicts P(home win) for each game from each team's recent form.

```
python mlp/01_build_dataset.py     # build pre-game features -> mlp/data/
python mlp/04_mlp.py               # run method 3 -> mlp/results/, mlp/figs/
```

- **Features:** for each team, the average over its previous 10 games of effective field goal %, turnover rate, offensive rebound rate, free throw rate, offensive and defensive rating (points scored and allowed per 100 possessions), possessions, and win %, plus days of rest and a back-to-back flag. Each feature is entered as home minus away. Windows are shifted by one game, so a game's own stats never enter its features, and they carry across seasons, so early-season games use the end of last season.
- **Home court:** every row is from the home team's perspective, so home-court advantage is learned by the network's bias term.
- **Model:** standardized inputs, ReLU hidden layers, sigmoid output, early stopping.
- **Tuning (beyond course content):** hidden layer sizes and L2 penalty α chosen by time-series cross-validation (each validation fold uses only games after its training data), scored on log loss. Selected: one hidden layer of 16 units, α = 1.0.
- **Train/test split:** train on 2021-22 to 2024-25, test on 2025-26. The 5 neutral-site test games are excluded because they have no true home team, leaving 1,225 test games.
- **Data:** `mlp/01_build_dataset.py` pulls the same LeagueGameLog data from nba_api and builds the rolling features. It does not use `data/games.csv`.
- **Results:** 65.4% accuracy, log loss 0.625, Brier 0.218, AUC 0.697 on 2025-26, vs. 55.4% for always picking the home team.

### Outputs (`mlp/`)

| File | Contents |
| --- | --- |
| `results/mlp_test.csv` | 2025-26 accuracy, log loss, Brier score and AUC for the MLP and the home-team baseline |
| `figs/mlp_confusion.png` | report Figure 5: test-season confusion matrix |
| `figs/mlp_calibration.png` | report Figure 6: test-season calibration curve |
| `data/raw_2021-22.csv` … `raw_2025-26.csv` | raw LeagueGameLog pulls, one file per season |
| `data/team_games.csv` | one row per team per game, with per-game efficiency stats and rolling features |
| `data/games_model.csv` | one row per game (home perspective) with `HOME_WIN` and the home-minus-away features used by the MLP |
