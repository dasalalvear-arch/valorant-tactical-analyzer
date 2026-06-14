import pandas as pd
import numpy as np

# Coordinate ranges per map (x_min, x_max, y_min, y_max)
# Derived from Riot API game coordinate observations (origin at 0)
MAP_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "ascent":   (0, 6300, 0, 4200),
    "bind":     (0, 7100, 0, 5100),
    "haven":    (0, 6600, 0, 4400),
    "split":    (0, 6500, 0, 4100),
    "fracture": (0, 6800, 0, 4600),
}
DEFAULT_BOUNDS = (0, 10000, 0, 10000)


def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x_norm"] = 0.0
    df["y_norm"] = 0.0

    for map_name, group_idx in df.groupby("map").groups.items():
        x_min, x_max, y_min, y_max = MAP_BOUNDS.get(map_name, DEFAULT_BOUNDS)
        df.loc[group_idx, "x_norm"] = (df.loc[group_idx, "x"] - x_min) / (x_max - x_min)
        df.loc[group_idx, "y_norm"] = (df.loc[group_idx, "y"] - y_min) / (y_max - y_min)

    df["x_norm"] = df["x_norm"].clip(0, 1)
    df["y_norm"] = df["y_norm"].clip(0, 1)
    return df


def assign_zones(df: pd.DataFrame, rows: int = 4, cols: int = 6) -> pd.DataFrame:
    df = normalize_coordinates(df)
    df["zone_row"] = (df["y_norm"] * rows).astype(int).clip(0, rows - 1)
    df["zone_col"] = (df["x_norm"] * cols).astype(int).clip(0, cols - 1)
    return df


def compute_zone_stats(
    df: pd.DataFrame,
    min_events: int = 10,
) -> pd.DataFrame:
    group_cols = ["player", "map", "zone_row", "zone_col", "side"]

    kills = df[df["result"] == "kill"].groupby(group_cols).size().reset_index(name="kills")
    deaths = df[df["result"] == "death"].groupby(group_cols).size().reset_index(name="deaths")

    stats = kills.merge(deaths, on=group_cols, how="outer").fillna(0)
    stats["kills"] = stats["kills"].astype(int)
    stats["deaths"] = stats["deaths"].astype(int)
    stats["total_events"] = stats["kills"] + stats["deaths"]
    stats["kill_rate"] = stats["kills"] / stats["total_events"].replace(0, np.nan)
    stats["kill_rate"] = stats["kill_rate"].fillna(0).round(4)
    stats["insufficient_sample"] = stats["total_events"] < min_events

    return stats
