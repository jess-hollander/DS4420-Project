"""
Step 1: pull NBA team game logs and build a leakage-free, game-level dataset.

Outputs:
  data/team_games.csv  one row per team per game (used by the Beta-Binomial and AR models)
  data/games_model.csv
  One row per game from the home team's perspective.
  Target: HOME_WIN (1 = home team won)
  Features: D_* = home team's pre-game value minus away team's pre-game value
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Five most recent complete seasons, matching the team's dataset.
# Train: 2021-22 to 2024-25. Test: 2025-26.
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
WINDOW = 10      # rolling window: last 10 games
MIN_GAMES = 5    # need at least 5 prior games before a team's row is usable
DATA = Path("data")
DATA.mkdir(exist_ok=True)

# Per-game stats that get turned into rolling pre-game features
STATS = ["ORTG", "DRTG", "POSS", "EFG", "TOV_PCT", "OREB_PCT", "FT_RATE", "WIN"]


def pull_season(season: str) -> pd.DataFrame:
    """Download one season of team game logs (cached to CSV after first pull)."""
    cache = DATA / f"raw_{season}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"GAME_ID": str})
    print(f"Pulling {season}...")
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        timeout=60,
    ).get_data_frames()[0]
    df["SEASON"] = season
    df.to_csv(cache, index=False)
    time.sleep(1.5)  # be polite to the NBA.com API
    return df


def add_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's box score to each team-game row."""
    cols = ["TEAM_ID", "PTS", "FGA", "FTA", "OREB", "DREB", "TOV"]
    opp = df[["GAME_ID"] + cols].rename(columns={c: f"OPP_{c}" for c in cols})
    merged = df.merge(opp, on="GAME_ID")
    return merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()


def game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game efficiency stats from the box score."""
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["WIN"] = (df["WL"] == "W").astype(int)

    poss = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    opp_poss = df["OPP_FGA"] - df["OPP_OREB"] + df["OPP_TOV"] + 0.44 * df["OPP_FTA"]
    df["POSS"] = (poss + opp_poss) / 2                          # pace proxy
    df["ORTG"] = 100 * df["PTS"] / df["POSS"]                   # points scored per 100 poss
    df["DRTG"] = 100 * df["OPP_PTS"] / df["POSS"]               # points allowed per 100 poss
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]      # effective FG%
    df["TOV_PCT"] = df["TOV"] / poss                            # turnovers per possession
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"]) # offensive rebound rate
    df["FT_RATE"] = df["FTA"] / df["FGA"]                       # free throw rate
    return df


def rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling averages over each team's previous WINDOW games.
    The window carries across seasons, so early-season games use the end of last
    season and every 2025-26 test game gets features (only early 2021-22 is dropped).
    shift(1) is the leakage guard: the current game is never in its own features.
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    g = df.groupby("TEAM_ID")
    for s in STATS:
        df[f"R_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        )
    # Days since the team's previous game, capped at 7 (known before tip-off)
    df["REST"] = g["GAME_DATE"].diff().dt.days.clip(upper=7).fillna(7)
    df["B2B"] = (df["REST"] == 1).astype(int)
    return df


def to_games(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse two team rows per game into one home-perspective row."""
    feats = [f"R_{s}" for s in STATS] + ["REST", "B2B"]
    home = df[df["HOME"] == 1][["GAME_ID", "GAME_DATE", "SEASON",
                                "TEAM_ABBREVIATION", "WIN"] + feats]
    away = df[df["HOME"] == 0][["GAME_ID", "TEAM_ABBREVIATION"] + feats]
    games = home.merge(away, on="GAME_ID", suffixes=("_H", "_A"))
    games = games.rename(columns={"WIN": "HOME_WIN"})

    for f in feats:
        games[f"D_{f}"] = games[f"{f}_H"] - games[f"{f}_A"]

    d_cols = [c for c in games.columns if c.startswith("D_")]
    games = games.dropna(subset=d_cols)  # drops each season's first few games
    return games.sort_values("GAME_DATE").reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.concat([pull_season(s) for s in SEASONS], ignore_index=True)
    df = game_stats(add_opponent(raw))
    df = rolling_features(df)
    df.to_csv(DATA / "team_games.csv", index=False)   # team-game level (Beta-Binomial, AR)
    games = to_games(df)                               # game level, home perspective (MLP)
    games.to_csv(DATA / "games_model.csv", index=False)

    print(f"Saved {len(games):,} games to data/games_model.csv")
    print(f"Home win rate: {games['HOME_WIN'].mean():.3f}")
    print(games.groupby("SEASON").size().rename("games"))
