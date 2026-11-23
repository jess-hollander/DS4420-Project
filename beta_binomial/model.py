"""
Beta-Binomial model for NBA team strength, written by hand.

We only use NumPy, pandas and the standard library here (no scikit-learn,
statsmodels, PyMC, scipy.stats, etc.). The Bayesian updates, the empirical
Bayes prior, the posterior predictive and the evaluation metrics are all
coded ourselves.

Model
-----
Every team i has some hidden strength theta_i, which we define as the
probability that team i beats an average team on a neutral court. If we
treat each game as a Bernoulli trial we get:

    theta_i            ~ Beta(alpha_i, beta_i)          (prior)
    W_i | theta_i, n_i ~ Binomial(n_i, theta_i)         (likelihood)
    theta_i | W_i      ~ Beta(alpha_i + W_i, beta_i + L_i)  (conjugate posterior)

Prior:
    * The prior for the whole league, Beta(alpha0, beta0), comes from
      empirical Bayes. We pick the values that maximize the Beta-Binomial
      marginal likelihood of the team-season records in the training seasons,
      using a simple grid search.
    * Season carry-over (something extra we added beyond the class material):
      at the start of a new season, a team's prior is the league prior plus a
      scaled down version of its record from last season:
          alpha_i = alpha0 + rho * W_prev,  beta_i = beta0 + rho * L_prev
      rho is between 0 and 1 and gets tuned on the training seasons. With
      rho = 0 last season is ignored, and with rho = 1 it counts fully.

Home court advantage:
    eta = P(home team wins | the two teams are equal) ~ Beta(a_h, b_h). This
    gets updated as the season goes using every game that wasn't at a
    neutral site.

Predicting a game (similar to log5 / Bradley-Terry):
    logit P(home wins) = logit(theta_home) - logit(theta_away) + logit(eta)
    For neutral-site games we leave out the eta term. To get the posterior
    predictive P(home wins | data) we sample from the three Beta posteriors
    and average, so the uncertainty in the parameters is included in the
    prediction instead of just plugging in point estimates.

We go through the games one date at a time and only use games from earlier
dates to make each prediction, so the model never sees a game's result
before predicting it.
"""

import math

import numpy as np
import pandas as pd

EPS = 1e-6


# ---------------------------------------------------------------------------
# Math helper functions
# ---------------------------------------------------------------------------
_lgamma = np.vectorize(math.lgamma, otypes=[float])


def log_beta_fn(a, b):
    """Log of the beta function: log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)."""
    return _lgamma(a) + _lgamma(b) - _lgamma(np.asarray(a) + np.asarray(b))


def log_choose(n, k):
    """Log of n choose k."""
    n, k = np.asarray(n, float), np.asarray(k, float)
    return _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Empirical Bayes prior
# ---------------------------------------------------------------------------
def beta_binomial_loglik(wins, games, alpha, beta):
    """
    Marginal log likelihood of the win totals under a Beta-Binomial model.
    For one team this is:
        p(W | n, a, b) = C(n, W) * B(W + a, n - W + b) / B(a, b)
    and we sum the logs over all teams.
    """
    wins, games = np.asarray(wins, float), np.asarray(games, float)
    return np.sum(log_choose(games, wins)
                  + log_beta_fn(wins + alpha, games - wins + beta)
                  - log_beta_fn(alpha, beta))


def fit_prior_empirical_bayes(wins, games,
                              mean_grid=np.linspace(0.40, 0.60, 81),
                              strength_grid=np.linspace(1.0, 100.0, 991)):
    """
    Fit Beta(alpha0, beta0) to the team-season records by trying every
    combination of mean and strength on a grid and keeping the one with the
    highest Beta-Binomial marginal likelihood. Here
    alpha0 = mean * strength and beta0 = (1 - mean) * strength.

    You can think of the strength (alpha0 + beta0) as the number of fake
    "pseudo-games" at a .500 record that every team starts the season with.

    Returns a dict with alpha0, beta0, mean, strength and loglik. It also
    includes a method of moments estimate so we can compare the two.
    """
    wins, games = np.asarray(wins, float), np.asarray(games, float)
    best = (-np.inf, None, None)
    for m in mean_grid:
        a = m * strength_grid
        b = (1 - m) * strength_grid
        ll = np.array([beta_binomial_loglik(wins, games, ai, bi) for ai, bi in zip(a, b)])
        j = int(np.argmax(ll))
        if ll[j] > best[0]:
            best = (ll[j], m, strength_grid[j])
    ll, m, s = best

    # Method of moments check, using Var(W/n) = mu(1-mu)/n + Var(theta)(1 - 1/n)
    p = wins / games
    n = games.mean()
    mu = p.mean()
    var_theta = max(p.var(ddof=1) - mu * (1 - mu) / n, 1e-6) / (1 - 1 / n)
    s_mom = mu * (1 - mu) / var_theta - 1

    return {"alpha0": m * s, "beta0": (1 - m) * s, "mean": m, "strength": s,
            "loglik": ll, "mom_mean": mu, "mom_strength": s_mom}


def team_season_records(games):
    """Get the final number of wins and games played for each team in each season."""
    home = pd.DataFrame({"SEASON": games["SEASON"], "TEAM": games["HOME_TEAM_ABBREVIATION"],
                         "WIN": games["HOME_WIN"]})
    away = pd.DataFrame({"SEASON": games["SEASON"], "TEAM": games["AWAY_TEAM_ABBREVIATION"],
                         "WIN": 1 - games["HOME_WIN"]})
    both = pd.concat([home, away])
    return both.groupby(["SEASON", "TEAM"])["WIN"].agg(wins="sum", games="size").reset_index()


# ---------------------------------------------------------------------------
# Sequential Beta-Binomial model
# ---------------------------------------------------------------------------
class BetaBinomialNBA:
    """
    Bayesian team rating model that updates after every game date.

    Parameters
    ----------
    alpha0, beta0 : the league-wide Beta prior on team strength
    rho           : how much of last season's record carries over (0 to 1)
    home_prior    : (a, b) for the Beta prior on the home court win chance eta
    n_samples     : number of Monte Carlo samples for the posterior predictive
                    (if 0, we just use the posterior means instead)
    seed          : random seed so results are reproducible
    """

    def __init__(self, alpha0, beta0, rho=0.0, home_prior=(1.0, 1.0),
                 n_samples=2000, seed=0):
        self.alpha0, self.beta0 = float(alpha0), float(beta0)
        self.rho = float(rho)
        self.home_prior = tuple(float(x) for x in home_prior)
        self.n_samples = int(n_samples)
        self.seed = seed

    # Predict all the games on one date using the current posteriors
    def _predict_batch(self, a_h, b_h, a_a, b_a, a_eta, b_eta, neutral, rng):
        if self.n_samples <= 0:
            th_h = a_h / (a_h + b_h)
            th_a = a_a / (a_a + b_a)
            eta = a_eta / (a_eta + b_eta)
            z = logit(th_h) - logit(th_a) + (1 - neutral) * logit(eta)
            return sigmoid(z)

        S = self.n_samples
        th_h = rng.beta(a_h[:, None], b_h[:, None], size=(len(a_h), S))
        th_a = rng.beta(a_a[:, None], b_a[:, None], size=(len(a_a), S))
        eta = rng.beta(a_eta, b_eta, size=(1, S))
        z = logit(th_h) - logit(th_a) + (1 - neutral)[:, None] * logit(eta)
        return sigmoid(z).mean(axis=1)

    def run(self, games):
        """
        Go through the games in order by date. For each date we first predict
        the games and then update the posteriors with the results.

        Returns
        -------
        preds   : a copy of `games` with some added columns: P_HOME (the
                  posterior predictive), HOME_A/HOME_B/AWAY_A/AWAY_B (posterior
                  parameters before the game), HOME_W/HOME_L/AWAY_W/AWAY_L
                  (each team's record so far this season) and ETA_MEAN.
        history : DataFrame with every team's posterior after each game date
                  (SEASON, GAME_DATE, TEAM, ALPHA, BETA, W, L).
        """
        rng = np.random.default_rng(self.seed)
        g = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        teams = sorted(set(g["HOME_TEAM_ABBREVIATION"]) | set(g["AWAY_TEAM_ABBREVIATION"]))
        idx = {t: i for i, t in enumerate(teams)}
        T = len(teams)

        hi = g["HOME_TEAM_ABBREVIATION"].map(idx).to_numpy()
        ai = g["AWAY_TEAM_ABBREVIATION"].map(idx).to_numpy()
        y = g["HOME_WIN"].to_numpy()
        neutral = g["NEUTRAL"].to_numpy().astype(float)

        W = np.zeros(T)          # wins and losses so far this season
        L = np.zeros(T)
        prior_a = np.full(T, self.alpha0)
        prior_b = np.full(T, self.beta0)
        a_eta, b_eta = self.home_prior

        out = {k: np.zeros(len(g)) for k in
               ["P_HOME", "HOME_A", "HOME_B", "AWAY_A", "AWAY_B",
                "HOME_W", "HOME_L", "AWAY_W", "AWAY_L", "ETA_MEAN"]}
        history = []
        season = None

        for date, rows in g.groupby("GAME_DATE", sort=True).indices.items():
            rows = np.asarray(rows)
            s = g.at[rows[0], "SEASON"]
            if s != season:                       # new season, so reset with the carry-over prior
                if season is not None:
                    prior_a = self.alpha0 + self.rho * W
                    prior_b = self.beta0 + self.rho * L
                W[:] = 0.0
                L[:] = 0.0
                season = s

            post_a = prior_a + W
            post_b = prior_b + L
            h, a = hi[rows], ai[rows]

            out["P_HOME"][rows] = self._predict_batch(post_a[h], post_b[h], post_a[a], post_b[a],
                                                      a_eta, b_eta, neutral[rows], rng)
            out["HOME_A"][rows], out["HOME_B"][rows] = post_a[h], post_b[h]
            out["AWAY_A"][rows], out["AWAY_B"][rows] = post_a[a], post_b[a]
            out["HOME_W"][rows], out["HOME_L"][rows] = W[h], L[h]
            out["AWAY_W"][rows], out["AWAY_L"][rows] = W[a], L[a]
            out["ETA_MEAN"][rows] = a_eta / (a_eta + b_eta)

            # Update the posteriors with today's results (conjugate update)
            yr = y[rows]
            np.add.at(W, h, yr)
            np.add.at(L, h, 1 - yr)
            np.add.at(W, a, 1 - yr)
            np.add.at(L, a, yr)
            nn = neutral[rows] == 0
            a_eta += yr[nn].sum()
            b_eta += (1 - yr[nn]).sum()

            history.append(pd.DataFrame({"SEASON": s, "GAME_DATE": date, "TEAM": teams,
                                         "ALPHA": prior_a + W, "BETA": prior_b + L,
                                         "W": W.copy(), "L": L.copy()}))

        preds = g.copy()
        for k, v in out.items():
            preds[k] = v
        self.final_eta_ = (a_eta, b_eta)
        return preds, pd.concat(history, ignore_index=True)


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------
def beta_credible_interval(a, b, level=0.95, n_draws=200_000, seed=0):
    """Credible interval for Beta(a, b), found by sampling and taking quantiles from each tail."""
    rng = np.random.default_rng(seed)
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    ci = np.empty((len(a), 2))
    for i in range(len(a)):
        d = rng.beta(a[i], b[i], size=n_draws)
        ci[i] = np.quantile(d, [lo, hi])
    return ci


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def accuracy(y, p):
    return float(np.mean((np.asarray(p) >= 0.5) == (np.asarray(y) == 1)))


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def evaluate(y, p):
    return {"accuracy": accuracy(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p)}


def calibration_table(y, p, n_bins=10):
    """Put the predictions into bins and compare the average prediction to how often the home team actually won."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = [{"bin_low": edges[k], "bin_high": edges[k + 1], "n": int((b == k).sum()),
             "mean_pred": p[b == k].mean(), "obs_rate": y[b == k].mean()}
            for k in range(n_bins) if (b == k).any()]
    return pd.DataFrame(rows)
