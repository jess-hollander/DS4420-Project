# DS 4420 Project: Predicting NBA Game Outcomes

Jessica Hollander, Suhana Singh Madia, Rayd Hussain

A team's win-loss record doesn't always tell you how good they actually are,
especially early in the season when they've only played a handful of games.
For this project we try to estimate team strength and predict who wins each
game using three different methods:

| # | Method | Implementation | Owner | Status |
|---|--------|----------------|-------|--------|
| 1 | Beta-Binomial (Bayesian) | manual (just NumPy) | Jessica | done, see `beta_binomial/` |
| 2 | AR(p) on team scoring | statsmodels | Suhana | TODO |
| 3 | MLP win/loss classifier | scikit-learn | Rayd | TODO |

## Setup

```bash
pip install -r requirements.txt
python data/fetch_nba_data.py      # downloads and cleans the data, makes data/games.csv
python -m beta_binomial.run        # runs method 1, saves stuff to results/beta_binomial/
```

## Data (`data/`)

We got the data from [`nba_api`](https://github.com/swar/nba_api), which pulls
from the official NBA.com stats site. We used the `LeagueGameLog` endpoint for
the regular seasons from 2021-22 through 2025-26.

* `raw_team_game_logs.csv`: 12,300 rows. Each game shows up twice, once for
  each team, with that team's box score stats.
* `games.csv`: 6,150 rows, one per game. It has `GAME_ID, SEASON, GAME_DATE,
  NEUTRAL`, the team IDs, abbreviations and box score stats for both sides
  (`HOME_*` and `AWAY_*`), plus `HOME_WIN` and `POINT_DIFF`. All three methods
  use this file as their input.

How we cleaned it: the `MATCHUP` column says "vs." for the home team and "@"
for the away team, so we used that to figure out who was home. Then we joined
the two rows for each game on `GAME_ID` and checked that every game had exactly
one winner. There were 10 neutral-site games (international games and NBA Cup
semifinals) where both teams were listed with "@". We kept those but set
`NEUTRAL = 1`, and just called whichever team comes first alphabetically the
"home" team.

## Method 1: Beta-Binomial (`beta_binomial/`)

The model is written by hand using NumPy, without scikit-learn, statsmodels,
scipy.stats or PyMC. The docstring at the top of `beta_binomial/model.py`
explains the math in more detail.

* **Team strength:** θᵢ is the probability that team i beats an average team
  on a neutral court, with θᵢ ~ Beta(α, β). Since each game is basically a
  Bernoulli trial, after W wins and L losses the posterior is just
  Beta(α + W, β + L) (conjugate update).
* **Prior (empirical Bayes):** we pick Beta(α₀, β₀) by maximizing the
  Beta-Binomial marginal likelihood of the team-season records in the training
  data. We did this with a grid search we wrote ourselves.
* **Carrying over last season** (this goes a bit beyond what we covered in
  class): at the start of a new season the prior becomes α₀ + ρ·W_prev and
  β₀ + ρ·L_prev. We tuned ρ on the training seasons.
* **Home court:** η also gets a Beta prior and is updated as we go using every
  game that wasn't at a neutral site.
* **Prediction:** logit P(home win) = logit θ_home − logit θ_away + logit η.
  We draw 2,000 samples from the posteriors and average the result to get the
  posterior predictive. Games are predicted one day at a time, and each
  prediction only uses games from before that day.
* **Train/test split:** we train on 2021-22 to 2024-25 (this is where the prior
  and ρ come from) and test on 2025-26.

### Outputs (`results/beta_binomial/`)

| File | What's in it |
|------|--------------|
| `test_metrics.csv` | accuracy, log loss and Brier score on 2025-26 for our model, the baselines and the ablations |
| `accuracy_by_stage.csv` | test accuracy broken up by how far into the season it is |
| `shrinkage_analysis.csv` | whether the posterior mean or raw win % does better at predicting a team's win % for the rest of the season |
| `rho_tuning.csv` | training log loss for each value of ρ we tried |
| `posteriors_2025-26.csv` | each team's posterior and 95% credible interval at the end of the season |
| `calibration_beta_binomial.csv` | calibration bins for the test season |
| `predictions.csv` | the pre-game posterior and predicted P(home win) for every game |
| `summary.json` | the prior, ρ, the home court posterior and the test metrics |
| `fig1_posterior_credible_intervals.png` | team strength posteriors at the end of 2025-26 |
| `fig2_calibration.png` | how well calibrated the predicted probabilities are on the test season |
| `fig3_posterior_evolution.png` | posterior vs. raw win % over the season for 3 teams |
