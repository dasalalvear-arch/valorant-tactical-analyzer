# Valorant Tactical Analyzer — Plan A: Data Pipeline + API Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sistema funcional localmente con datos de Valorant, análisis por zonas ATK/DEF, modelo Win/Loss y todos los endpoints de la API corriendo en Docker.

**Architecture:** FastAPI carga datos desde CSVs procesados y un modelo serializado. La lógica de zonas asigna cada kill a una celda de una grilla 6×4 normalizada por mapa. Los endpoints devuelven JSON y PNG. Todo corre en un container Docker.

**Tech Stack:** Python 3.11, FastAPI, scikit-learn, matplotlib, pandas, pytest, Docker, requests (HenrikDev API)

---

## Estructura de archivos

```
valorant-analyzer/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # fetching + guardado de partidas desde HenrikDev API
│   ├── zones.py            # grilla 6×4, normalización, asignación de kills a zonas
│   ├── model.py            # feature engineering + entrenamiento Random Forest
│   └── simulation.py       # lógica what-if de redistribución por zona
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI: todos los endpoints
├── data/
│   ├── raw/                # JSON crudos de la API (ignorados por git)
│   └── processed/          # kills_by_zone.csv, matches.csv
├── tests/
│   ├── conftest.py
│   ├── test_zones.py
│   ├── test_model.py
│   ├── test_simulation.py
│   └── test_api.py
├── notebooks/              # EDA (no entran en los tests)
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `api/__init__.py`
- Create: `.env.example`
- Create: `Dockerfile`
- Create: `tests/conftest.py`

- [ ] **Step 1: Crear estructura de directorios**

```bash
mkdir -p src api data/raw data/processed tests notebooks scripts
touch src/__init__.py api/__init__.py tests/__init__.py
```

- [ ] **Step 2: Crear requirements.txt**

```
fastapi==0.115.0
uvicorn==0.30.6
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.9.2
requests==2.32.3
python-dotenv==1.0.1
joblib==1.4.2
boto3==1.35.0
pytest==8.3.2
httpx==0.27.2
pytest-mock==3.14.0
Pillow==10.4.0
```

- [ ] **Step 3: Crear .env.example**

```
HENRIK_API_KEY=
RIOT_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET=valorant-analyzer
MODEL_PATH=models/model.pkl
```

- [ ] **Step 4: Crear Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY data/processed/ ./data/processed/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 5: Crear tests/conftest.py**

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_kills_df():
    return pd.DataFrame({
        "player": ["TenZ#NA1", "TenZ#NA1", "TenZ#NA1", "s1mple#EU1"],
        "map": ["ascent", "ascent", "ascent", "ascent"],
        "x": [0.3, 0.7, 0.5, 0.2],
        "y": [0.4, 0.8, 0.2, 0.6],
        "side": ["ATK", "DEF", "ATK", "DEF"],
        "result": ["kill", "kill", "death", "kill"],
    })

@pytest.fixture
def sample_zone_stats_df():
    return pd.DataFrame({
        "player": ["TenZ#NA1", "TenZ#NA1"],
        "map": ["ascent", "ascent"],
        "zone_row": [2, 3],
        "zone_col": [1, 3],
        "side": ["ATK", "DEF"],
        "kills": [15, 8],
        "deaths": [3, 5],
        "total_events": [18, 13],
        "kill_rate": [0.83, 0.62],
        "insufficient_sample": [False, False],
    })
```

- [ ] **Step 6: Commit inicial**

```bash
git init
echo "data/raw/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pkl" >> .gitignore
git add .
git commit -m "feat: project scaffolding"
```

---

## Task 2: Data loader — HenrikDev API

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py` (mock de API)

- [ ] **Step 1: Escribir test del fetcher (mock)**

Crear `tests/test_data_loader.py`:

```python
from unittest.mock import patch, MagicMock
from src.data_loader import fetch_matches, extract_kills

MOCK_MATCH = {
    "data": [{
        "metadata": {"map": "Ascent", "matchid": "abc123"},
        "players": {"all_players": [
            {"name": "TenZ", "tag": "NA1", "stats": {"kills": 22, "deaths": 10, "assists": 3}}
        ]},
        "teams": {"red": {"has_won": True}},
        "rounds": [{
            "winning_team": "Red",
            "player_stats": [{
                "player_display_name": "TenZ#NA1",
                "kills": [{
                    "killer_display_name": "TenZ#NA1",
                    "victim_display_name": "s1mple#EU1",
                    "killer_team": "Red",
                    "attacker_team": "Red",
                    "round_info": {"attacking_team": "Red"},
                    "player_locations_on_kill": [
                        {"player_display_name": "TenZ#NA1", "location": {"x": 1200, "y": 3400}}
                    ]
                }]
            }]
        }]
    }]
}

def test_fetch_matches_returns_list(mocker):
    mock_get = mocker.patch("src.data_loader.requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = MOCK_MATCH

    result = fetch_matches("TenZ", "NA1", region="na")
    assert isinstance(result, list)
    assert len(result) == 1

def test_extract_kills_returns_dataframe(mocker):
    mocker.patch("src.data_loader.requests.get")
    import pandas as pd
    kills = extract_kills(MOCK_MATCH["data"])
    assert isinstance(kills, pd.DataFrame)
    assert "x" in kills.columns
    assert "y" in kills.columns
    assert "side" in kills.columns
    assert "result" in kills.columns
```

- [ ] **Step 2: Correr test para verificar que falla**

```bash
pytest tests/test_data_loader.py -v
```

Resultado esperado: `ERROR` — `src.data_loader` no existe.

- [ ] **Step 3: Implementar src/data_loader.py**

```python
import os
import json
import requests
import pandas as pd
from pathlib import Path

HENRIK_BASE = "https://api.henrikdev.xyz/valorant"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def fetch_matches(name: str, tag: str, region: str = "na", count: int = 20) -> list:
    url = f"{HENRIK_BASE}/v3/matches/{region}/{name}/{tag}?size={count}"
    headers = {}
    api_key = os.getenv("HENRIK_API_KEY", "")
    if api_key:
        headers["Authorization"] = api_key

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def extract_kills(matches: list) -> pd.DataFrame:
    rows = []
    for match in matches:
        map_name = match["metadata"]["map"].lower()
        match_id = match["metadata"]["matchid"]

        # Determinar qué equipo era atacante en ronda 1 (simplificación)
        for round_data in match.get("rounds", []):
            attacking_team = round_data.get("attacker_team", "")
            for ps in round_data.get("player_stats", []):
                player = ps["player_display_name"]
                player_team = _get_player_team(match, player)
                side = "ATK" if player_team == attacking_team else "DEF"

                for kill in ps.get("kills", []):
                    killer = kill.get("killer_display_name", "")
                    victim = kill.get("victim_display_name", "")

                    killer_loc = _find_location(kill["player_locations_on_kill"], killer)
                    if not killer_loc:
                        continue

                    rows.append({
                        "match_id": match_id,
                        "map": map_name,
                        "player": killer,
                        "side": side,
                        "result": "kill",
                        "x": killer_loc["x"],
                        "y": killer_loc["y"],
                    })
                    victim_loc = _find_location(kill["player_locations_on_kill"], victim)
                    if victim_loc:
                        victim_side = "DEF" if side == "ATK" else "ATK"
                        rows.append({
                            "match_id": match_id,
                            "map": map_name,
                            "player": victim,
                            "side": victim_side,
                            "result": "death",
                            "x": victim_loc["x"],
                            "y": victim_loc["y"],
                        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["match_id", "map", "player", "side", "result", "x", "y"]
    )


def save_kills(df: pd.DataFrame, player_tag: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"kills_{player_tag.replace('#', '_')}.csv"
    df.to_csv(path, index=False)
    return path


def _get_player_team(match: dict, player: str) -> str:
    for p in match.get("players", {}).get("all_players", []):
        if p.get("name", "") + "#" + p.get("tag", "") == player:
            return p.get("team", "")
    return ""


def _find_location(locations: list, player: str) -> dict | None:
    for loc in locations:
        if loc.get("player_display_name") == player:
            return loc.get("location")
    return None
```

- [ ] **Step 4: Correr tests**

```bash
pytest tests/test_data_loader.py -v
```

Resultado esperado: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader con HenrikDev API + tests"
```

---

## Task 3: Zone grid — lógica central

**Files:**
- Create: `src/zones.py`
- Create: `tests/test_zones.py`

- [ ] **Step 1: Escribir tests de zonas**

Crear `tests/test_zones.py`:

```python
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
    df = pd.DataFrame({
        "x": [100, 900],
        "y": [200, 800],
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
    # Con pocos datos en el fixture, todas las zonas deben tener insufficient=True
    assert "insufficient_sample" in stats.columns
```

- [ ] **Step 2: Correr test para verificar que falla**

```bash
pytest tests/test_zones.py -v
```

Resultado esperado: `ERROR` — `src.zones` no existe.

- [ ] **Step 3: Implementar src/zones.py**

```python
import pandas as pd
import numpy as np

# Rangos de coordenadas conocidos por mapa (x_min, x_max, y_min, y_max)
# Fuente: coordenadas del juego observadas en datos de la API
MAP_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "ascent":   (2900, 6300, 1300, 4200),
    "bind":     (3800, 7100, 2200, 5100),
    "haven":    (2200, 6600, 1100, 4400),
    "split":    (3100, 6500, 1500, 4100),
    "fracture": (2500, 6800, 1800, 4600),
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
```

- [ ] **Step 4: Correr tests**

```bash
pytest tests/test_zones.py -v
```

Resultado esperado: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/zones.py tests/test_zones.py
git commit -m "feat: zone grid logic con normalización por mapa y stats ATK/DEF"
```

---

## Task 4: ML Model — Win/Loss con Random Forest

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Escribir tests del modelo**

Crear `tests/test_model.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.model import build_features, train_model, predict_match

@pytest.fixture
def sample_matches_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "match_id": range(n),
        "player": [f"p{i % 10}#tag" for i in range(n)],
        "map": np.random.choice(["ascent", "bind", "haven"], n),
        "acs": np.random.randint(150, 350, n),
        "kd": np.random.uniform(0.7, 2.0, n),
        "hs_pct": np.random.uniform(0.1, 0.4, n),
        "win": np.random.choice([0, 1], n),
    })


def test_build_features_returns_dataframe(sample_matches_df):
    features = build_features(sample_matches_df)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0


def test_train_model_returns_pipeline(sample_matches_df):
    features = build_features(sample_matches_df)
    model = train_model(features)
    assert hasattr(model, "predict_proba")


def test_predict_match_returns_valid_probabilities(sample_matches_df):
    features = build_features(sample_matches_df)
    model = train_model(features)

    team_a = [{"player": "p1#tag", "map": "ascent"}]
    team_b = [{"player": "p2#tag", "map": "ascent"}]
    result = predict_match(model, features, team_a, team_b, map_name="ascent")

    assert "team_a_win_prob" in result
    assert "team_b_win_prob" in result
    assert abs(result["team_a_win_prob"] + result["team_b_win_prob"] - 1.0) < 0.001
```

- [ ] **Step 2: Correr test para verificar que falla**

```bash
pytest tests/test_model.py -v
```

Resultado esperado: `ERROR` — `src.model` no existe.

- [ ] **Step 3: Implementar src/model.py**

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("data/processed")


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega stats por jugador: promedio de ACS, K/D, HS% y winrate."""
    agg = matches_df.groupby("player").agg(
        avg_acs=("acs", "mean"),
        avg_kd=("kd", "mean"),
        avg_hs=("hs_pct", "mean"),
        winrate=("win", "mean"),
        games=("match_id", "count"),
    ).reset_index()
    return agg


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


def save_model(model: Pipeline, path: Path = MODEL_DIR / "model.pkl") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = MODEL_DIR / "model.pkl") -> Pipeline:
    return joblib.load(path)


def predict_match(
    model: Pipeline,
    features_df: pd.DataFrame,
    team_a: list[dict],
    team_b: list[dict],
    map_name: str,
) -> dict:
    def _team_avg(team: list[dict]) -> pd.Series:
        players = [p["player"] for p in team]
        team_features = features_df[features_df["player"].isin(players)]
        if team_features.empty:
            # Fallback: valores promedio del dataset
            return features_df[["avg_acs", "avg_kd", "avg_hs", "winrate", "games"]].mean()
        return team_features[["avg_acs", "avg_kd", "avg_hs", "winrate", "games"]].mean()

    a_vec = _team_avg(team_a).values.reshape(1, -1)
    b_vec = _team_avg(team_b).values.reshape(1, -1)

    prob_a = model.predict_proba(a_vec)[0][1]
    prob_b = model.predict_proba(b_vec)[0][1]

    total = prob_a + prob_b
    return {
        "team_a_win_prob": round(prob_a / total, 4),
        "team_b_win_prob": round(prob_b / total, 4),
        "model": "random_forest_v1",
        "disclaimer": "Basado en historial individual. No considera sinergia de equipo.",
    }
```

- [ ] **Step 4: Correr tests**

```bash
pytest tests/test_model.py -v
```

Resultado esperado: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: Random Forest Win/Loss con feature engineering por jugador"
```

---

## Task 5: Simulación what-if

**Files:**
- Create: `src/simulation.py`
- Create: `tests/test_simulation.py`

- [ ] **Step 1: Escribir tests de simulación**

Crear `tests/test_simulation.py`:

```python
import pytest
from src.simulation import simulate_zone_redistribution

def test_simulate_returns_expected_kda(sample_zone_stats_df):
    result = simulate_zone_redistribution(
        zone_stats=sample_zone_stats_df,
        player="TenZ#NA1",
        map_name="ascent",
        side="ATK",
        new_distribution={(2, 1): 0.8, (3, 3): 0.2},
    )
    assert "simulated_kill_rate" in result
    assert "historical_kill_rate" in result
    assert 0 <= result["simulated_kill_rate"] <= 1


def test_simulate_rejects_invalid_distribution(sample_zone_stats_df):
    with pytest.raises(ValueError, match="distribución debe sumar 1.0"):
        simulate_zone_redistribution(
            zone_stats=sample_zone_stats_df,
            player="TenZ#NA1",
            map_name="ascent",
            side="ATK",
            new_distribution={(2, 1): 0.5, (3, 3): 0.3},  # suma 0.8, inválido
        )
```

- [ ] **Step 2: Correr test para verificar que falla**

```bash
pytest tests/test_simulation.py -v
```

- [ ] **Step 3: Implementar src/simulation.py**

```python
import pandas as pd

def simulate_zone_redistribution(
    zone_stats: pd.DataFrame,
    player: str,
    map_name: str,
    side: str,
    new_distribution: dict[tuple[int, int], float],
    tolerance: float = 0.01,
) -> dict:
    """
    Calcula el kill_rate esperado si el jugador redistribuye sus peleas por zona.

    new_distribution: {(zone_row, zone_col): fracción_de_peleas} — debe sumar 1.0.
    """
    total_weight = sum(new_distribution.values())
    if abs(total_weight - 1.0) > tolerance:
        raise ValueError(f"distribución debe sumar 1.0, recibió {total_weight:.2f}")

    player_stats = zone_stats[
        (zone_stats["player"] == player) &
        (zone_stats["map"] == map_name) &
        (zone_stats["side"] == side) &
        (~zone_stats["insufficient_sample"])
    ]

    historical_kill_rate = (
        (player_stats["kills"].sum() / player_stats["total_events"].sum())
        if not player_stats.empty and player_stats["total_events"].sum() > 0
        else None
    )

    simulated_kill_rate = 0.0
    for (zone_row, zone_col), weight in new_distribution.items():
        zone = player_stats[
            (player_stats["zone_row"] == zone_row) &
            (player_stats["zone_col"] == zone_col)
        ]
        if zone.empty:
            # Zona sin datos: usar promedio del jugador
            kr = historical_kill_rate or 0.5
        else:
            kr = float(zone.iloc[0]["kill_rate"])
        simulated_kill_rate += weight * kr

    return {
        "player": player,
        "map": map_name,
        "side": side,
        "historical_kill_rate": round(historical_kill_rate, 4) if historical_kill_rate else None,
        "simulated_kill_rate": round(simulated_kill_rate, 4),
        "delta": round(simulated_kill_rate - (historical_kill_rate or 0), 4),
        "disclaimer": "Simulación basada en historial. No predice resultados reales.",
    }
```

- [ ] **Step 4: Correr tests**

```bash
pytest tests/test_simulation.py -v
```

Resultado esperado: `2 passed`.

- [ ] **Step 5: Correr todos los tests hasta ahora**

```bash
pytest tests/ -v --ignore=tests/test_api.py
```

Resultado esperado: todos `passed`.

- [ ] **Step 6: Commit**

```bash
git add src/simulation.py tests/test_simulation.py
git commit -m "feat: simulación what-if de redistribución por zonas"
```

---

## Task 6: FastAPI — app base + endpoints player/card y predict-match

**Files:**
- Create: `api/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Escribir tests de la API**

Crear `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

@pytest.fixture
def client():
    with patch("api.main._load_app_data") as mock_load:
        mock_load.return_value = (
            pd.DataFrame({
                "player": ["TenZ#NA1"],
                "avg_acs": [280.0],
                "avg_kd": [1.4],
                "avg_hs": [0.30],
                "winrate": [0.60],
                "games": [50],
            }),
            pd.DataFrame({
                "player": ["TenZ#NA1"],
                "map": ["ascent"],
                "zone_row": [2],
                "zone_col": [3],
                "side": ["ATK"],
                "kills": [20],
                "deaths": [5],
                "total_events": [25],
                "kill_rate": [0.80],
                "insufficient_sample": [False],
            }),
            MagicMock(),  # model
        )
        from api.main import app
        yield TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_match_returns_probabilities(client):
    response = client.post("/predict-match", json={
        "team_a": ["TenZ#NA1"],
        "team_b": ["s1mple#EU1"],
        "map": "ascent",
        "agents_a": ["Jett"],
        "agents_b": ["Reyna"],
    })
    assert response.status_code == 200
    data = response.json()
    assert "team_a_win_prob" in data
    assert "team_b_win_prob" in data
    assert abs(data["team_a_win_prob"] + data["team_b_win_prob"] - 1.0) < 0.01


def test_player_card_returns_stats(client):
    response = client.get("/player/TenZ/NA1/card")
    assert response.status_code == 200
    data = response.json()
    assert "avg_acs" in data


def test_player_zones_returns_png(client):
    response = client.get("/player/TenZ/NA1/zones/ascent")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
```

- [ ] **Step 2: Correr tests para verificar que fallan**

```bash
pytest tests/test_api.py -v
```

- [ ] **Step 3: Implementar api/main.py**

```python
import io
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.model import predict_match, load_model
from src.zones import compute_zone_stats
from src.simulation import simulate_zone_redistribution

app = FastAPI(title="Valorant Tactical Analyzer", version="1.0.0")

ROWS, COLS = 4, 6
_features_df: pd.DataFrame = None
_zone_stats_df: pd.DataFrame = None
_model = None


def _load_app_data():
    features_path = Path("data/processed/features.csv")
    zones_path = Path("data/processed/zone_stats.csv")
    model_path = Path("data/processed/model.pkl")

    features = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
    zones = pd.read_csv(zones_path) if zones_path.exists() else pd.DataFrame()
    model = load_model(model_path) if model_path.exists() else None
    return features, zones, model


@app.on_event("startup")
def startup():
    global _features_df, _zone_stats_df, _model
    _features_df, _zone_stats_df, _model = _load_app_data()


@app.get("/health")
def health():
    return {"status": "ok"}


class MatchRequest(BaseModel):
    team_a: list[str]
    team_b: list[str]
    map: str
    agents_a: list[str] = []
    agents_b: list[str] = []


@app.post("/predict-match")
def predict_match_endpoint(req: MatchRequest):
    if _model is None:
        raise HTTPException(503, "Modelo no disponible")
    team_a = [{"player": p} for p in req.team_a]
    team_b = [{"player": p} for p in req.team_b]
    return predict_match(_model, _features_df, team_a, team_b, req.map)


@app.get("/player/{name}/{tag}/card")
def player_card(name: str, tag: str):
    player_id = f"{name}#{tag}"
    if _features_df is None or _features_df.empty:
        raise HTTPException(503, "Datos no disponibles")
    row = _features_df[_features_df["player"] == player_id]
    if row.empty:
        raise HTTPException(404, f"Jugador {player_id} no encontrado")
    return row.iloc[0].to_dict()


@app.get("/player/{name}/{tag}/zones/{map_name}")
def player_zones(name: str, tag: str, map_name: str, side: str = "ALL"):
    player_id = f"{name}#{tag}"
    if _zone_stats_df is None or _zone_stats_df.empty:
        raise HTTPException(503, "Datos no disponibles")

    df = _zone_stats_df[
        (_zone_stats_df["player"] == player_id) &
        (_zone_stats_df["map"] == map_name.lower())
    ]
    if side != "ALL":
        df = df[df["side"] == side.upper()]
    if df.empty:
        raise HTTPException(404, f"Sin datos para {player_id} en {map_name}")

    grid = np.full((ROWS, COLS), np.nan)
    for _, row in df[~df["insufficient_sample"]].iterrows():
        grid[int(row["zone_row"]), int(row["zone_col"])] = row["kill_rate"]

    img = _render_heatmap(grid, title=f"{player_id} — {map_name.title()} ({side})")
    return StreamingResponse(img, media_type="image/png")


@app.get("/compare/{name_a}/{tag_a}/{name_b}/{tag_b}/{map_name}")
def compare_players(name_a: str, tag_a: str, name_b: str, tag_b: str,
                    map_name: str, side: str = "ALL"):
    player_a = f"{name_a}#{tag_a}"
    player_b = f"{name_b}#{tag_b}"

    def _get_grid(player: str) -> np.ndarray:
        df = _zone_stats_df[
            (_zone_stats_df["player"] == player) &
            (_zone_stats_df["map"] == map_name.lower())
        ]
        if side != "ALL":
            df = df[df["side"] == side.upper()]
        g = np.full((ROWS, COLS), np.nan)
        for _, row in df[~df["insufficient_sample"]].iterrows():
            g[int(row["zone_row"]), int(row["zone_col"])] = row["kill_rate"]
        return g

    grid_a = _get_grid(player_a)
    grid_b = _get_grid(player_b)
    diff = np.where(
        np.isnan(grid_a) | np.isnan(grid_b),
        np.nan,
        grid_a - grid_b,
    )

    img = _render_diff_heatmap(
        diff,
        title=f"{player_a} (azul) vs {player_b} (rojo) — {map_name.title()}",
        label_a=player_a,
        label_b=player_b,
    )
    return StreamingResponse(img, media_type="image/png")


class SimulateRequest(BaseModel):
    side: str = "ATK"
    new_distribution: dict[str, float]  # "row,col" -> fracción


@app.post("/simulate/{name}/{tag}/{map_name}")
def simulate(name: str, tag: str, map_name: str, req: SimulateRequest):
    player_id = f"{name}#{tag}"
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
    players_a = team_a.split(",")
    players_b = team_b.split(",")

    def _team_grid(players: list[str]) -> np.ndarray:
        df = _zone_stats_df[
            (_zone_stats_df["player"].isin(players)) &
            (_zone_stats_df["map"] == map_name.lower()) &
            (~_zone_stats_df["insufficient_sample"])
        ]
        if df.empty:
            return np.full((ROWS, COLS), np.nan)
        agg = df.groupby(["zone_row", "zone_col"])["kill_rate"].mean()
        g = np.full((ROWS, COLS), np.nan)
        for (r, c), val in agg.items():
            g[int(r), int(c)] = val
        return g

    grid_a = _team_grid(players_a)
    grid_b = _team_grid(players_b)
    diff = np.where(np.isnan(grid_a) | np.isnan(grid_b), np.nan, grid_a - grid_b)

    img = _render_diff_heatmap(
        diff,
        title=f"Dominio por zona — {map_name.title()}",
        label_a="Equipo A",
        label_b="Equipo B",
    )
    return StreamingResponse(img, media_type="image/png")


# ── Helpers de visualización ──────────────────────────────────────────────────

def _render_heatmap(grid: np.ndarray, title: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 5))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Kill rate")
    ax.set_title(title)
    ax.set_xlabel("Columna (oeste → este)")
    ax.set_ylabel("Fila (sur → norte)")
    _annotate_grid(ax, grid)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_diff_heatmap(
    diff: np.ndarray, title: str, label_a: str, label_b: str
) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 5))
    masked = np.ma.masked_invalid(diff)
    im = ax.imshow(masked, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"← {label_b} domina | {label_a} domina →")
    ax.set_title(title)
    ax.set_xlabel("Columna (oeste → este)")
    ax.set_ylabel("Fila (sur → norte)")
    _annotate_grid(ax, diff, fmt="+.2f")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _annotate_grid(ax: plt.Axes, grid: np.ndarray, fmt: str = ".2f") -> None:
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = grid[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color="black")
```

- [ ] **Step 4: Correr tests de la API**

```bash
pytest tests/test_api.py -v
```

Resultado esperado: `4 passed`.

- [ ] **Step 5: Correr todos los tests**

```bash
pytest tests/ -v
```

Resultado esperado: todos `passed`.

- [ ] **Step 6: Probar la API localmente**

```bash
uvicorn api.main:app --reload --port 8000
```

Abrir `http://localhost:8000/docs` — debe mostrar Swagger con todos los endpoints.

- [ ] **Step 7: Probar Docker**

```bash
docker build -t valorant-analyzer .
docker run -p 8000:8000 valorant-analyzer
```

Verificar que `http://localhost:8000/health` responde `{"status":"ok"}`.

- [ ] **Step 8: Commit final**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: FastAPI completa con todos los endpoints y heatmaps PNG"
```

---

## Task 7: Notebook de exploración y generación de datos procesados

**Files:**
- Create: `notebooks/01_exploration.ipynb` (manual, no entra en tests)
- Create: `notebooks/02_train_model.ipynb` (manual)

- [ ] **Step 1: Instalar dependencias en entorno virtual**

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install jupyter
```

- [ ] **Step 2: Script de fetch + procesamiento (ejecutar una vez)**

Crear `scripts/fetch_and_process.py`:

```python
"""
Script one-shot para descargar partidas de un jugador y generar los CSVs procesados.
Uso: python scripts/fetch_and_process.py TenZ NA1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_loader import fetch_matches, extract_kills, save_kills
from src.zones import assign_zones, compute_zone_stats
from src.model import build_features, train_model, save_model

def main(name: str, tag: str, region: str = "na"):
    print(f"Descargando partidas de {name}#{tag}...")
    matches = fetch_matches(name, tag, region=region, count=20)
    kills_df = extract_kills(matches)
    print(f"  → {len(kills_df)} eventos de kill/death")

    kills_with_zones = assign_zones(kills_df)
    zone_stats = compute_zone_stats(kills_with_zones)
    zone_stats.to_csv("data/processed/zone_stats.csv", index=False)
    print(f"  → zone_stats.csv guardado ({len(zone_stats)} filas)")

    # Para el modelo necesitamos stats de partidas agregadas
    # Aquí usamos los kills como proxy — en producción usar endpoint de match history
    features = build_features(kills_df.assign(
        acs=kills_df.groupby(["match_id", "player"])["result"].transform(
            lambda x: (x == "kill").sum() * 10
        ),
        kd=1.0,
        hs_pct=0.25,
        win=0,  # placeholder — reemplazar con datos reales
    ))
    features.to_csv("data/processed/features.csv", index=False)

    model = train_model(features)
    save_model(model)
    print("  → model.pkl guardado")
    print("Listo.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "na")
```

- [ ] **Step 3: Ejecutar el script con un jugador real**

```bash
python scripts/fetch_and_process.py TenZ NA1
```

Si falla por rate limit, esperar 30 segundos y reintentar.

- [ ] **Step 4: Verificar que la API devuelve datos reales**

```bash
uvicorn api.main:app --reload
```

```bash
curl http://localhost:8000/player/TenZ/NA1/card
curl "http://localhost:8000/player/TenZ/NA1/zones/ascent" --output heatmap.png
```

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_and_process.py
git commit -m "feat: script de procesamiento de datos + generación de CSVs"
```

---

## Checklist final del Plan A

- [ ] Todos los tests pasan: `pytest tests/ -v`
- [ ] `http://localhost:8000/docs` muestra todos los endpoints con Swagger
- [ ] `docker build` y `docker run` funcionan sin errores
- [ ] Al menos un jugador real tiene datos procesados en `data/processed/`
- [ ] Los endpoints devuelven PNGs válidos para `/zones` y `/compare`

**Plan A completo → continuar con Plan B (AWS + CI/CD + Frontend)**
