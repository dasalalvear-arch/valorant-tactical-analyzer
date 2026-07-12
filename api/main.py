"""
API FastAPI del Valorant Tactical Analyzer.

Carga en el arranque (lifespan) tres artefactos ya procesados desde
data/processed: features de jugadores, zone_stats y el modelo Win/Loss.
Los endpoints devuelven JSON (card, predict, simulate) o PNG (heatmaps por
zona). No entrena nada ni llama a la API externa: consume lo que generó el
pipeline offline (ver scripts/fetch_and_process.py).
"""
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin display: renderiza a buffer, no a ventana
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.model import predict_match, load_model
from src.simulation import simulate_zone_redistribution

# Grilla 6×4 — debe coincidir con la usada al generar zone_stats (src/zones.py).
ROWS, COLS = 4, 6
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Globals poblados por el lifespan al arrancar. None = artefacto ausente en disco.
_features_df: pd.DataFrame | None = None
_zone_stats_df: pd.DataFrame | None = None
_model = None


def _load_app_data():
    """Lee features/zone_stats/modelo de disco. Devuelve None por artefacto ausente."""
    features_path = PROCESSED_DIR / "features.csv"
    zones_path = PROCESSED_DIR / "zone_stats.csv"
    model_path = PROCESSED_DIR / "model.pkl"

    features = pd.read_csv(features_path) if features_path.exists() else None
    zones = pd.read_csv(zones_path) if zones_path.exists() else None
    model = load_model(model_path) if model_path.exists() else None
    return features, zones, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _features_df, _zone_stats_df, _model
    _features_df, _zone_stats_df, _model = _load_app_data()
    yield


app = FastAPI(title="Valorant Tactical Analyzer", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


class MatchRequest(BaseModel):
    team_a: list[str]
    team_b: list[str]
    map: str


@app.post("/predict-match")
def predict_match_endpoint(req: MatchRequest):
    if _model is None or _features_df is None:
        raise HTTPException(503, "Modelo no disponible")
    team_a = [{"player": p} for p in req.team_a]
    team_b = [{"player": p} for p in req.team_b]
    return predict_match(_model, _features_df, team_a, team_b, req.map)


@app.get("/player/{name}/{tag}/card")
def player_card(name: str, tag: str):
    if _features_df is None:
        raise HTTPException(503, "Datos no disponibles")
    player_id = f"{name}#{tag}"
    row = _features_df[_features_df["player"] == player_id]
    if row.empty:
        raise HTTPException(404, f"Jugador {player_id} no encontrado")
    return row.iloc[0].to_dict()


@app.get("/player/{name}/{tag}/zones/{map_name}")
def player_zones(name: str, tag: str, map_name: str, side: str = "ALL"):
    if _zone_stats_df is None:
        raise HTTPException(503, "Datos no disponibles")
    player_id = f"{name}#{tag}"
    df = _zone_stats_df[
        (_zone_stats_df["player"] == player_id)
        & (_zone_stats_df["map"] == map_name.lower())
    ]
    if side != "ALL":
        df = df[df["side"] == side.upper()]
    if df.empty:
        raise HTTPException(404, f"Sin datos para {player_id} en {map_name}")

    grid = _kill_rate_grid(df)
    img = _render_heatmap(grid, title=f"{player_id} — {map_name.title()} ({side})")
    return StreamingResponse(img, media_type="image/png")


@app.get("/compare/{name_a}/{tag_a}/{name_b}/{tag_b}/{map_name}")
def compare_players(name_a: str, tag_a: str, name_b: str, tag_b: str,
                    map_name: str, side: str = "ALL"):
    if _zone_stats_df is None:
        raise HTTPException(503, "Datos no disponibles")
    player_a = f"{name_a}#{tag_a}"
    player_b = f"{name_b}#{tag_b}"

    grid_a = _player_grid(player_a, map_name, side)
    grid_b = _player_grid(player_b, map_name, side)
    # Solo comparamos celdas con datos en AMBOS jugadores; el resto queda NaN.
    diff = np.where(np.isnan(grid_a) | np.isnan(grid_b), np.nan, grid_a - grid_b)

    img = _render_diff_heatmap(
        diff,
        title=f"{player_a} vs {player_b} — {map_name.title()}",
        label_a=player_a,
        label_b=player_b,
    )
    return StreamingResponse(img, media_type="image/png")


class SimulateRequest(BaseModel):
    side: str = "ATK"
    new_distribution: dict[str, float]  # "row,col" -> fracción de peleas


@app.post("/simulate/{name}/{tag}/{map_name}")
def simulate(name: str, tag: str, map_name: str, req: SimulateRequest):
    if _zone_stats_df is None:
        raise HTTPException(503, "Datos no disponibles")
    player_id = f"{name}#{tag}"
    # El JSON no admite tuplas como clave: llegan "row,col" y las reconvertimos.
    distribution = {
        tuple(int(x) for x in k.split(",")): v
        for k, v in req.new_distribution.items()
    }
    try:
        return simulate_zone_redistribution(
            _zone_stats_df, player_id, map_name.lower(), req.side, distribution
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/team-zones/{team_a}/{team_b}/{map_name}")
def team_zones(team_a: str, team_b: str, map_name: str):
    if _zone_stats_df is None:
        raise HTTPException(503, "Datos no disponibles")
    grid_a = _team_grid(team_a.split(","), map_name)
    grid_b = _team_grid(team_b.split(","), map_name)
    diff = np.where(np.isnan(grid_a) | np.isnan(grid_b), np.nan, grid_a - grid_b)

    img = _render_diff_heatmap(
        diff,
        title=f"Dominio por zona — {map_name.title()}",
        label_a="Equipo A",
        label_b="Equipo B",
    )
    return StreamingResponse(img, media_type="image/png")


# ── Helpers de grilla ─────────────────────────────────────────────────────────

def _kill_rate_grid(df: pd.DataFrame) -> np.ndarray:
    """Vuelca kill_rate de filas zone_stats a una grilla ROWS×COLS (NaN si no hay dato)."""
    grid = np.full((ROWS, COLS), np.nan)
    for _, row in df[~df["insufficient_sample"]].iterrows():
        grid[int(row["zone_row"]), int(row["zone_col"])] = row["kill_rate"]
    return grid


def _player_grid(player: str, map_name: str, side: str) -> np.ndarray:
    df = _zone_stats_df[
        (_zone_stats_df["player"] == player)
        & (_zone_stats_df["map"] == map_name.lower())
    ]
    if side != "ALL":
        df = df[df["side"] == side.upper()]
    return _kill_rate_grid(df)


def _team_grid(players: list[str], map_name: str) -> np.ndarray:
    """Grilla del equipo: promedio de kill_rate por celda entre sus jugadores."""
    df = _zone_stats_df[
        (_zone_stats_df["player"].isin(players))
        & (_zone_stats_df["map"] == map_name.lower())
        & (~_zone_stats_df["insufficient_sample"])
    ]
    grid = np.full((ROWS, COLS), np.nan)
    if df.empty:
        return grid
    for (r, c), val in df.groupby(["zone_row", "zone_col"])["kill_rate"].mean().items():
        grid[int(r), int(c)] = val
    return grid


# ── Helpers de visualización ──────────────────────────────────────────────────

def _render_heatmap(grid: np.ndarray, title: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Kill rate")
    ax.set_title(title)
    ax.set_xlabel("Columna (oeste → este)")
    ax.set_ylabel("Fila (sur → norte)")
    _annotate_grid(ax, grid)
    return _to_png(fig)


def _render_diff_heatmap(diff: np.ndarray, title: str, label_a: str, label_b: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.ma.masked_invalid(diff), cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"← {label_b} domina | {label_a} domina →")
    ax.set_title(title)
    ax.set_xlabel("Columna (oeste → este)")
    ax.set_ylabel("Fila (sur → norte)")
    _annotate_grid(ax, diff, fmt="+.2f")
    return _to_png(fig)


def _annotate_grid(ax, grid: np.ndarray, fmt: str = ".2f") -> None:
    """Escribe el valor numérico en cada celda con dato."""
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = grid[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color="black")


def _to_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)  # liberar la figura: sin esto matplotlib acumula memoria por request
    buf.seek(0)
    return buf
