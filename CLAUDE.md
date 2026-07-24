# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML system that analyzes Valorant matches with a focus on spatial (per-zone) analysis of the map. Built as a portfolio project (see `docs/superpowers/specs/2026-06-13-valorant-tactical-analyzer-design.md` for the full design doc and `docs/superpowers/plans/2026-06-13-plan-A-data-api-core.md` for the implementation plan). Comments and docstrings in this codebase are written in Spanish — match that convention when editing `src/`.

Current state: `src/` (data loader, zones, model, simulation) is implemented and tested. `api/main.py` (the FastAPI app) is implemented and tested (`tests/test_api.py`); it serves the descriptive/zone/simulate/predict endpoints but consumes precomputed artifacts from `data/processed/` — the offline pipeline that generates them (`scripts/fetch_and_process.py`) does **not exist yet**, so the app returns 503 until those artifacts are present. The next planned feature — a predictive round/site win model (`LogisticRegression`, temporal split, per-zone coefficients) — is designed in `docs/superpowers/specs/2026-07-11-predictive-round-site-model-design.md` but **not yet implemented**.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # deps (prod + test)
pytest tests/ -v                                          # run all tests
pytest tests/test_zones.py -v                              # single test file
pytest tests/test_zones.py::test_assign_zones_produces_valid_cells -v  # single test
docker build -t valorant-analyzer .
docker run -p 8000:8000 valorant-analyzer
```

No linter/formatter config is present in the repo.

## Architecture

Two independent pipelines feeding one FastAPI layer:

1. **Spatial/zone pipeline** (`src/data_loader.py` → `src/zones.py` → `src/simulation.py`): fetches raw matches, flattens kills/deaths to (x, y, side, result) events, normalizes coordinates per-map into a 6×4 grid, aggregates kill_rate per player/map/zone/side, and supports "what-if" redistribution queries.
2. **Win/Loss model pipeline** (`src/model.py`): aggregates per-player match history (ACS, K/D, HS%, winrate) into features and trains a `RandomForestClassifier` (`class_weight="balanced"`). Note the target is derived from the player's own historical winrate (`>= 0.5`), not a per-match label — a known simplification for this demo, documented inline in `train_model`.

These two pipelines are independent: zone events (kills/deaths with coordinates) and match-level features (ACS/KD/winrate) come from different aggregations of the same underlying match data and don't share a DataFrame shape.

### `src/data_loader.py` — HenrikDev API ingestion
- Auth is via query param `api_key`, not a header (HenrikDev-specific).
- `.env` is loaded explicitly via `load_dotenv()` with an absolute path resolved from `Path(__file__).parent.parent` — works the same in local runs, tests, and Docker regardless of cwd.
- Only "tactical" modes (`competitive`, `unrated`, `swiftplay`, `premier`) are parsed — arcade modes (Deathmatch, Escalation, etc.) are skipped because they lack `player_locations_on_kill`/plant data needed for zone/side assignment.
- ATK/DEF side is inferred from spike *plant* events per round (`_round_attackers`): whichever team planted first is attacker until one halftime swap. This model does not handle overtime (side alternates every round there), so overtime rounds get an approximated side.
- Error messages from failed requests are redacted (`_redact`) to strip the API key out of any URL that leaks into an exception message — don't remove this when touching error handling.

### `src/zones.py` — the core spatial module
- ⚠️ **`MAP_BOUNDS` is confirmed WRONG against real API data (verified 2026-07-23).** It assumes positive ranges (ascent `x: 2900–6300, y: 1300–4200`), but real HenrikDev coordinates are negative and origin-centred (e.g. ascent `x≈-2786, y≈-5545`; observed span across two maps was `x: -3261→10114, y: -8698→6352`). Real events therefore get `clip(0,1)`-ed onto the edges — all 5 rows of a real sample collapsed to `x_norm=y_norm=0.0`. **`assign_zones` / `compute_zone_stats` output is unreliable for live data**, and the claim that five maps are "calibrated against real data" does not hold for any of them. Repairing this is deliberately deferred to its own piece of work (it cascades into `conftest.py`, `test_zones`, `test_simulation`, `test_api` and the predictive-model spec). Until then, for anything coordinate-related prefer the official transform in `src/mapviz.py`, which is validated against real data.
- Grid/threshold notes below still apply, but read them knowing the zone assignment feeding them is currently wrong for real matches. Unknown maps fall back to `DEFAULT_BOUNDS` and emit a `warnings.warn`.
- Grid is 6 cols × 4 rows (24 zones) — deliberately sized so most zones clear the minimum-sample threshold (see below); don't shrink it without checking sample sizes.
- `compute_zone_stats(min_events=10)` flags (does not drop) zone/side combos with `total_events < min_events` as `insufficient_sample` — callers must respect that flag rather than trusting `kill_rate` blindly.
- `compute_zone_stats` requires `assign_zones()` to have run first (validates required columns and raises `ValueError` otherwise).

### `src/simulation.py`
Pure function over already-computed `zone_stats` (no model training). `new_distribution` weights must sum to 1.0 (±`tolerance`); zones with no historical data fall back to the player's overall historical kill_rate (or 0.5 if that's also unavailable — checked via `is not None`, since a real 0.0 kill_rate must not be treated as "no data").

### `src/mapviz.py` + `src/mapplot.py` — overlay de kills sobre el mapa real
- `mapviz.py` is the pure core (no matplotlib). `get_map_asset(map)` downloads and caches the official minimap PNG plus the coordinate-transform constants from valorant-api.com — image at `data/maps/<map>.png` (gitignored), constants at `data/maps/transforms.json` (versioned, 13 maps, so a fresh clone needs no network for coordinates). `game_to_pixel(df, transform, size)` converts game coords to image pixels.
- The official transform **swaps axes** (the minimap's X uses the game's `y`) and `yMultiplier` is negative (vertical flip). Validated against real match data — sampled events land inside `[0,1]` and visually on real corridors and both bomb sites.
- **This path deliberately bypasses `MAP_BOUNDS`**, and is currently the only coordinate path that is correct for live data. See the warning in the `zones.py` section above.
- `mapplot.py` is the thin matplotlib layer: `plot_kills_on_map(kills, map, player, side, result, mode)` renders a scatter or a hexbin density map. **`result` matters:** `extract_kills` emits two rows per kill — a `kill` at the killer's position and a `death` at the victim's. With `result="ALL"` the scatter colours them separately (kills blue, deaths orange, with a legend) and the title says "eventos"; `"kill"` / `"death"` filter to one type and drop the legend. Blue/orange rather than green/red so the two stay distinguishable with red-green colour blindness.
- The heatmap always draws a colorbar — the hexbin gradient encodes *event count per cell* and is uninterpretable without a scale. Note `mincnt` hides cells below that count, which with sparse data can hide almost everything.
- `_encuadre(img)` crops the view to the minimap's real content: the valorant-api PNGs carry ~10% transparent padding (ascent's map occupies 949×889 of a 1024×1024 image). It only moves the viewport — point coordinates are untouched.

## Testing conventions

Shared fixtures (`sample_kills_df`, `sample_zone_stats_df`, `mock_model`) live in `tests/conftest.py` — use these rather than constructing ad hoc DataFrames when a test's shape matches. `sample_kills_df` uses coordinates matching `MAP_BOUNDS` for ascent — which, per the 2026-07-23 finding above, are **not** values real matches produce, so that fixture validates against fictional data. Rewriting it is part of the deferred `MAP_BOUNDS` work; don't treat its green tests as evidence the zone pipeline works on live data. API calls in tests are mocked via `pytest-mock` (`mocker.patch("src.data_loader.requests.get")`) — no real network calls in the test suite.
