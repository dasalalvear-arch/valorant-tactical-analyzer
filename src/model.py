import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).parent.parent / "data" / "processed"


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per-player stats from match history."""
    return matches_df.groupby("player").agg(
        avg_acs=("acs", "mean"),
        avg_kd=("kd", "mean"),
        avg_hs=("hs_pct", "mean"),
        winrate=("win", "mean"),
        games=("match_id", "count"),
    ).reset_index()


def train_model(features_df: pd.DataFrame) -> Pipeline:
    X = features_df[["avg_acs", "avg_kd", "avg_hs", "winrate", "games"]]
    y = (features_df["winrate"] >= 0.5).astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def save_model(model: Pipeline, path: Path = None) -> Path:
    if path is None:
        path = MODEL_DIR / "model.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = None) -> Pipeline:
    if path is None:
        path = MODEL_DIR / "model.pkl"
    return joblib.load(Path(path))


def predict_match(
    model: Pipeline,
    features_df: pd.DataFrame,
    team_a: list[dict],
    team_b: list[dict],
    map_name: str,
) -> dict:
    fallback = features_df[["avg_acs", "avg_kd", "avg_hs", "winrate", "games"]].mean()

    def _team_vector(team: list[dict]) -> np.ndarray:
        players = [p["player"] for p in team]
        rows = features_df[features_df["player"].isin(players)]
        if rows.empty:
            vec = fallback.values
        else:
            vec = rows[["avg_acs", "avg_kd", "avg_hs", "winrate", "games"]].mean().values
        return vec.reshape(1, -1)

    prob_a = model.predict_proba(_team_vector(team_a))[0][1]
    prob_b = model.predict_proba(_team_vector(team_b))[0][1]
    total = prob_a + prob_b

    return {
        "team_a_win_prob": round(float(prob_a / total), 4),
        "team_b_win_prob": round(float(prob_b / total), 4),
        "model": "random_forest_v1",
        "disclaimer": "Basado en historial individual. No considera sinergia de equipo.",
    }
