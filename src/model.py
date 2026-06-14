"""
Modelo de predicción Win/Loss con Random Forest.

A diferencia del resto del pipeline, este módulo NO trabaja con eventos de kill
sino con un historial de partidas a nivel jugador (acs, kd, hs_pct, win).
build_features resume ese historial y train_model entrena el clasificador.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).parent.parent / "data" / "processed"

# Las 5 features del modelo, en una constante para no repetir la lista en
# build_features, train_model y predict_match (deben coincidir siempre).
FEATURE_COLS = ["avg_acs", "avg_kd", "avg_hs", "winrate", "games"]


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Resume el historial de cada jugador a una fila: promedios + winrate + nº de partidas."""
    return matches_df.groupby("player").agg(
        avg_acs=("acs", "mean"),
        avg_kd=("kd", "mean"),
        avg_hs=("hs_pct", "mean"),
        winrate=("win", "mean"),
        games=("match_id", "count"),
    ).reset_index()


def train_model(features_df: pd.DataFrame) -> Pipeline:
    """Entrena un Pipeline StandardScaler + RandomForest."""
    X = features_df[FEATURE_COLS]
    # NOTA: el target se deriva del propio winrate (>= 0.5). Es una simplificación
    # para la demo; para un predictor real, el target debería ser el resultado de
    # una partida concreta, separado de las features previas a esa partida.
    y = (features_df["winrate"] >= 0.5).astype(int)

    # StandardScaler es necesario porque las features están en escalas muy distintas
    # (ACS en cientos, K/D en torno a 1): sin escalar, las grandes dominarían.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",  # compensa si hay más ganadores que perdedores
            random_state=42,          # reproducibilidad entre corridas
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def save_model(model: Pipeline, path: Path = None) -> Path:
    """Serializa el modelo a disco con joblib (default: data/processed/model.pkl)."""
    if path is None:
        path = MODEL_DIR / "model.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = None) -> Pipeline:
    """Carga el modelo serializado desde disco."""
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
    """Predice la probabilidad de victoria de cada equipo promediando a sus jugadores."""
    # Vector de respaldo: promedio del dataset, para jugadores sin historial propio.
    fallback = features_df[FEATURE_COLS].mean()

    def _team_vector(team: list[dict]) -> pd.DataFrame:
        players = [p["player"] for p in team]
        rows = features_df[features_df["player"].isin(players)]
        if rows.empty:
            vec = fallback.values
        else:
            vec = rows[FEATURE_COLS].mean().values
        # DataFrame con nombres de columna (no numpy array) para evitar los
        # warnings de sklearn por "features sin nombre".
        return pd.DataFrame([vec], columns=FEATURE_COLS)

    prob_a = model.predict_proba(_team_vector(team_a))[0][1]
    prob_b = model.predict_proba(_team_vector(team_b))[0][1]
    total = prob_a + prob_b

    # Guard contra división por cero si ambas probabilidades dieran 0.
    if total == 0.0:
        prob_a, prob_b = 0.5, 0.5
        total = 1.0

    # Normalizamos para que las dos probabilidades sumen 1 (es un duelo A vs B).
    return {
        "team_a_win_prob": round(float(prob_a / total), 4),
        "team_b_win_prob": round(float(prob_b / total), 4),
        "model": "random_forest_v1",
        "disclaimer": "Basado en historial individual. No considera sinergia de equipo.",
    }
