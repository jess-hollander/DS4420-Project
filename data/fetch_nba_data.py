"""
Downloads and cleans the data for our DS 4420 NBA project.

Gets the regular season team game logs for 2021-22 through 2025-26 from
NBA.com with the nba_api package (https://github.com/swar/nba_api). Then it
turns them into a table with one row per game that has the home team, the
away team and who won.

Files it saves (in data/):
    raw_team_game_logs.csv  - one row per team per game, just how the
                              LeagueGameLog endpoint gives it to us (we only
                              add a SEASON column)
    games.csv               - one row per game (home/away, scores, home_win).
                              All three of our methods use this file.

How to run it (from the main project folder):
    python data/fetch_nba_data.py            # uses the saved raw file if it's already there
    python data/fetch_nba_data.py --refresh  # downloads everything again
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = Path(__file__).resolve().parent
RAW_PATH = DATA_DIR / "raw_team_game_logs.csv"
GAMES_PATH = DATA_DIR / "games.csv"

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Box score stats we keep for both teams. The Beta-Binomial model only needs
# who won, but the AR and MLP methods will probably use these.
STAT_COLS = ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
             "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF"]


def fetch_season(season, retries=3, pause=1.5):
    """Download the team game logs for one season (one row per team per game)."""
    for attempt in range(1, retries + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="T",
                timeout=60,
            ).get_data_frames()[0]
            df["SEASON"] = season
            time.sleep(pause)  # wait a bit so we don't spam stats.nba.com
            return df
        except Exception as exc:  # this API times out a lot so we just retry
            print(f"  {season}: attempt {attempt} failed ({exc})")
            time.sleep(pause * attempt * 2)
    raise RuntimeError(f"Could not download season {season}")


def fetch_raw(refresh=False):
    """Load the raw team game logs, or download them if we don't have them saved yet."""
    if RAW_PATH.exists() and not refresh:
        print(f"Using cached raw data: {RAW_PATH.name}")
        return pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})

    frames = []
    for season in SEASONS:
        print(f"Downloading {season} ...")
        frames.append(fetch_season(season))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(raw):,} team-game rows -> {RAW_PATH.name}")
    return raw


def curate(raw):
    """
    Turn the team game rows into one row per game.

    The MATCHUP column looks like "AAA vs. BBB" for the home team and
    "AAA @ BBB" for the away team. We use that to split each GAME_ID into a
    home row and an away row and then join them back together.
    """
    raw = raw.copy()
    raw["GAME_ID"] = raw["GAME_ID"].astype(str).str.zfill(10)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw = raw.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])

    raw["IS_HOME"] = raw["MATCHUP"].str.contains("vs.", regex=False)

    # Neutral-site games (international games and the NBA Cup semifinals in
    # Las Vegas) have "@" for both teams. We mark these with NEUTRAL = 1 and
    # just call the team that comes first alphabetically the "home" team so
    # the game still ends up as one row. The models shouldn't give anyone
    # home court advantage in these games.
    n_home = raw.groupby("GAME_ID")["IS_HOME"].transform("sum")
    raw["NEUTRAL"] = (n_home == 0).astype(int)
    first_abbr = raw.groupby("GAME_ID")["TEAM_ABBREVIATION"].transform("min")
    neutral = raw["NEUTRAL"] == 1
    raw.loc[neutral, "IS_HOME"] = raw.loc[neutral, "TEAM_ABBREVIATION"] == first_abbr[neutral]

    shared = ("GAME_ID", "SEASON", "GAME_DATE", "NEUTRAL")
    keep = list(shared) + ["TEAM_ID", "TEAM_ABBREVIATION", "WL"] + STAT_COLS
    home = raw.loc[raw["IS_HOME"], keep]
    away = raw.loc[~raw["IS_HOME"], keep]

    home = home.rename(columns={c: f"HOME_{c}" for c in keep if c not in shared})
    away = away.rename(columns={c: f"AWAY_{c}" for c in keep if c not in shared})
    away = away.drop(columns=["SEASON", "GAME_DATE", "NEUTRAL"])

    games = home.merge(away, on="GAME_ID", how="inner", validate="one_to_one")

    # Quick checks to make sure every game has one winner and no team is playing itself
    games = games[games["HOME_WL"].isin(["W", "L"]) & games["AWAY_WL"].isin(["W", "L"])]
    assert (games["HOME_WL"] != games["AWAY_WL"]).all(), "game with two winners/losers"
    assert (games["HOME_TEAM_ID"] != games["AWAY_TEAM_ID"]).all()

    games["HOME_WIN"] = (games["HOME_WL"] == "W").astype(int)
    games["POINT_DIFF"] = games["HOME_PTS"] - games["AWAY_PTS"]
    games = games.drop(columns=["HOME_WL", "AWAY_WL"])

    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return games


def load_games():
    """Load games.csv. If it doesn't exist yet, run the whole pipeline first to make it."""
    if not GAMES_PATH.exists():
        main([])
    return pd.read_csv(GAMES_PATH, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download raw data")
    args = parser.parse_args(argv)

    raw = fetch_raw(refresh=args.refresh)
    games = curate(raw)
    games.to_csv(GAMES_PATH, index=False)

    print(f"Saved {len(games):,} games -> {GAMES_PATH.name}")
    summary = games.groupby("SEASON").agg(games=("GAME_ID", "size"),
                                          home_win_rate=("HOME_WIN", "mean"))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
