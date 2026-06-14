import pandas as pd
import numpy as np
import pytest
from src.model import build_features, train_model, predict_match, save_model, load_model


@pytest.fixture
def sample_matches_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "match_id": [f"match_{i}" for i in range(n)],
        "player": [f"p{i % 10}#tag" for i in range(n)],
        "map": np.random.choice(["ascent", "bind", "haven"], n),
        "acs": np.random.randint(150, 350, n).astype(float),
        "kd": np.random.uniform(0.7, 2.0, n),
        "hs_pct": np.random.uniform(0.1, 0.4, n),
        "win": np.random.choice([0, 1], n),
    })


def test_build_features_returns_dataframe(sample_matches_df):
    features = build_features(sample_matches_df)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    assert "player" in features.columns
    assert "avg_acs" in features.columns
    assert "avg_kd" in features.columns
    assert "winrate" in features.columns


def test_build_features_aggregates_per_player(sample_matches_df):
    features = build_features(sample_matches_df)
    # 100 rows, 10 unique players → 10 feature rows
    assert len(features) == 10


def test_train_model_returns_pipeline(sample_matches_df):
    features = build_features(sample_matches_df)
    model = train_model(features)
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "predict")


def test_predict_match_probabilities_sum_to_one(sample_matches_df):
    features = build_features(sample_matches_df)
    model = train_model(features)
    team_a = [{"player": "p1#tag"}]
    team_b = [{"player": "p2#tag"}]
    result = predict_match(model, features, team_a, team_b, map_name="ascent")
    assert "team_a_win_prob" in result
    assert "team_b_win_prob" in result
    assert abs(result["team_a_win_prob"] + result["team_b_win_prob"] - 1.0) < 0.001


def test_predict_match_with_unknown_player_uses_fallback(sample_matches_df):
    features = build_features(sample_matches_df)
    model = train_model(features)
    team_a = [{"player": "unknown#player"}]
    team_b = [{"player": "p2#tag"}]
    result = predict_match(model, features, team_a, team_b, map_name="ascent")
    # Should not raise — unknown player falls back to dataset average
    assert result["team_a_win_prob"] >= 0
    assert result["team_b_win_prob"] >= 0


def test_save_and_load_model_roundtrip(tmp_path, sample_matches_df):
    features = build_features(sample_matches_df)
    model = train_model(features)
    path = save_model(model, tmp_path / "model.pkl")
    loaded = load_model(path)
    assert hasattr(loaded, "predict_proba")
    # Predictions should be identical after round-trip
    team_a = [{"player": "p1#tag"}]
    team_b = [{"player": "p2#tag"}]
    original = predict_match(model, features, team_a, team_b, "ascent")
    from_disk = predict_match(loaded, features, team_a, team_b, "ascent")
    assert original["team_a_win_prob"] == from_disk["team_a_win_prob"]
