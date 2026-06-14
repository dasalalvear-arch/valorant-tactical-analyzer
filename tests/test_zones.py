import pandas as pd
import pytest
from src.zones import normalize_coordinates, assign_zones, compute_zone_stats

ROWS, COLS = 4, 6


def test_normalize_coordinates_clamps_to_unit():
    df = pd.DataFrame({"x": [100, 500, 900], "y": [200, 600, 1000], "map": ["ascent"] * 3})
    result = normalize_coordinates(df)
    assert result["x_norm"].between(0, 1).all()
    assert result["y_norm"].between(0, 1).all()


def test_normalize_uses_per_map_bounds():
    # x=3200 → (3200-2900)/(6300-2900) ≈ 0.088
    # x=5800 → (5800-2900)/(6300-2900) ≈ 0.853
    df = pd.DataFrame({
        "x": [3200, 5800],
        "y": [2000, 3800],
        "map": ["ascent", "ascent"],
    })
    result = normalize_coordinates(df)
    assert result.loc[0, "x_norm"] < result.loc[1, "x_norm"]


def test_assign_zones_produces_valid_cells(sample_kills_df):
    df = assign_zones(sample_kills_df, rows=ROWS, cols=COLS)
    assert "zone_row" in df.columns
    assert "zone_col" in df.columns
    assert df["zone_row"].between(0, ROWS - 1).all()
    assert df["zone_col"].between(0, COLS - 1).all()


def test_compute_zone_stats_calculates_kill_rate(sample_kills_df):
    df_with_zones = assign_zones(sample_kills_df, rows=ROWS, cols=COLS)
    stats = compute_zone_stats(df_with_zones)
    assert "kill_rate" in stats.columns
    assert stats["kill_rate"].between(0, 1).all()


def test_compute_zone_stats_separates_atk_def(sample_kills_df):
    df_with_zones = assign_zones(sample_kills_df, rows=ROWS, cols=COLS)
    stats = compute_zone_stats(df_with_zones)
    sides = stats["side"].unique()
    assert "ATK" in sides or "DEF" in sides


def test_zones_with_insufficient_sample_flagged(sample_kills_df):
    df_with_zones = assign_zones(sample_kills_df, rows=ROWS, cols=COLS)
    stats = compute_zone_stats(df_with_zones, min_events=10)
    assert "insufficient_sample" in stats.columns
    # fixture has very few rows so all zones should be flagged as insufficient
    assert stats["insufficient_sample"].all()
