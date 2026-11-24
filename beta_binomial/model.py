"""
Beta-Binomial model of NBA team strength, implemented manually with NumPy.

Each team's strength theta ~ Beta(alpha, beta) is updated after every game
(posterior Beta(alpha + W, beta + L)), and games are predicted with
logit P(home win) = logit(theta_home) - logit(theta_away) + logit(eta).
"""

import math

import numpy as np
import pandas as pd

EPS = 1e-6


# Math helpers
_lgamma = np.vectorize(math.lgamma, otypes=[float])


def log_beta_fn(a, b):
    """Log of the beta function B(a, b)."""
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


# Empirical Bayes prior
def beta_binomial_loglik(wins, games, alpha, beta):
    """Beta-Binomial marginal log likelihood of the win totals."""
    wins, games = np.asarray(wins, float), np.asarray(games, float)
    return np.sum(log_choose(games, wins)
                  + log_beta_fn(wins + alpha, games - wins + beta)
                  - log_beta_fn(alpha, beta))


def fit_prior_empirical_bayes(wins, games,
                              mean_grid=np.linspace(0.40, 0.60, 81),
                              strength_grid=np.linspace(1.0, 100.0, 991)):
    """Fit the league prior Beta(alpha0, beta0) by grid search on the marginal likelihood."""
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

    # Method of moments estimate for comparison
    p = wins / games
    n = games.mean()
    mu = p.mean()
    var_theta = max(p.var(ddof=1) - mu * (1 - mu) / n, 1e-6) / (1 - 1 / n)
    s_mom = mu * (1 - mu) / var_theta - 1

    return {"alpha0": m * s, "beta0": (1 - m) * s, "mean": m, "strength": s,
            "loglik": ll, "mom_mean": mu, "mom_strength": s_mom}


def team_season_records(games):
    """Wins and games played for each team-season."""
    home = pd.DataFrame({"SEASON": games["SEASON"], "TEAM": games["HOME_TEAM_ABBREVIATION"],
                         "WIN": games["HOME_WIN"]})
    away = pd.DataFrame({"SEASON": games["SEASON"], "TEAM": games["AWAY_TEAM_ABBREVIATION"],
                         "WIN": 1 - games["HOME_WIN"]})
    both = pd.concat([home, away])
    return both.groupby(["SEASON", "TEAM"])["WIN"].agg(wins="sum", games="size").reset_index()


# Sequential Beta-Binomial model
class BetaBinomialNBA:
    """Team strength model updated after every game date (rho = season carry-over weight)."""

    def __init__(self, alpha0, beta0, rho=0.0, home_prior=(1.0, 1.0),
                 n_samples=2000, seed=0):
        self.alpha0, self.beta0 = float(alpha0), float(beta0)
        self.rho = float(rho)
        self.home_prior = tuple(float(x) for x in home_prior)
        self.n_samples = int(n_samples)
        self.seed = seed

    # Predict one date's games (Monte Carlo, or posterior means if n_samples = 0)
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
        """Predict each date's games, then update; returns (predictions, posterior history)."""
        rng = np.random.default_rng(self.seed)
        g = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        teams = sorted(set(g["HOME_TEAM_ABBREVIATION"]) | set(g["AWAY_TEAM_ABBREVIATION"]))
        idx = {t: i for i, t in enumerate(teams)}
        T = len(teams)

        hi = g["HOME_TEAM_ABBREVIATION"].map(idx).to_numpy()
        ai = g["AWAY_TEAM_ABBREVIATION"].map(idx).to_numpy()
        y = g["HOME_WIN"].to_numpy()
        neutral = g["NEUTRAL"].to_numpy().astype(float)

        W = np.zeros(T)          # current-season record
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
            if s != season:                       # new season: carry-over prior
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

            # Conjugate update with today's results
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


# Posterior summaries
def beta_credible_interval(a, b, level=0.95, n_draws=200_000, seed=0):
    """Equal-tailed credible interval for Beta(a, b) by sampling."""
    rng = np.random.default_rng(seed)
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    ci = np.empty((len(a), 2))
    for i in range(len(a)):
        d = rng.beta(a[i], b[i], size=n_draws)
        ci[i] = np.quantile(d, [lo, hi])
    return ci


# Evaluation metrics
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
    """Mean predicted vs. observed home win rate per probability bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = [{"bin_low": edges[k], "bin_high": edges[k + 1], "n": int((b == k).sum()),
             "mean_pred": p[b == k].mean(), "obs_rate": y[b == k].mean()}
            for k in range(n_bins) if (b == k).any()]
    return pd.DataFrame(rows)
