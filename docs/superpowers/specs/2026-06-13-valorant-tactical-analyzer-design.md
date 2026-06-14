# Design: Valorant Tactical Analyzer

**Fecha:** 2026-06-13  
**Autor:** David Salomón Alvear Luengo  
**Estado:** Aprobado

---

## Objetivo

Sistema de ML end-to-end que analiza partidas de Valorant con foco en análisis espacial por zonas del mapa. Construido para demostrar capacidades de ML Engineer en el CV: API desplegada en producción, pipeline CI/CD automatizado, y análisis táctico con datos reales.

---

## Por qué Valorant y no fútbol

Los proyectos de ML con fútbol (StatsBomb) son comunes en portafolios universitarios. El análisis espacial de esports con Riot API + AWS es original y permite hablar del dominio con autoridad en entrevistas. La arquitectura y el enfoque de ML son equivalentes; la diferencia está en la fuente de datos y el dominio.

---

## Stack

| Capa | Tecnología |
|------|-----------|
| Datos | HenrikDev API (prototipo) → Riot Official API (producción) |
| ML | scikit-learn — Random Forest Classifier |
| Backend | FastAPI |
| Visualización | matplotlib (heatmaps PNG) |
| Contenedor | Docker |
| Registro de imágenes | AWS ECR |
| Compute | AWS EC2 (t2.micro, free tier) |
| Almacenamiento | AWS S3 (datos procesados + model.pkl) |
| CI/CD | GitHub Actions |
| Frontend | HTML/JS simple (React en v2 si sobra tiempo) |

---

## Fuentes de datos

### Fase de prototipo: HenrikDev API
- URL: `api.henrikdev.xyz/valorant/v3/matches/{region}/{name}/{tag}`
- Sin autenticación requerida, ideal para desarrollo inicial
- Riesgo: API no oficial, puede cambiar sin aviso → plan B es Riot API oficial

### Fase de producción: Riot Official API
- Requiere dev key gratuita (aprobación en minutos en developer.riotgames.com)
- Endpoint clave: `GET /val/match/v1/matches/{matchId}`
- Incluye `roundResults[].playerStats[].kills[].playerLocations` — coordenadas X/Y de todos los jugadores en el momento de cada kill
- Rate limit: 20 req/s (dev key) → mitigar con caché en S3

### Opcional: VCT Datasets (Kaggle)
- Riot liberó datos de partidas profesionales del VCT
- Sin rate limits, útil para el modelo de predicción de partidas

---

## Arquitectura

```
LOCAL                     CI/CD                        AWS
─────                     ─────                        ───
HenrikDev/Riot API  →  Notebooks (EDA)
Riot API            →  src/ (zones, model)  →  S3 (data/raw, data/processed, models/)
                                                          ↓
git push → main  →  GitHub Actions  →  docker build
                                    →  docker push → ECR
                                    →  SSH → EC2 → pull + restart
                                                          ↓
                                                EC2 (Docker container)
                                                FastAPI — carga model.pkl desde S3
                                                          ↓
                                                  Cliente / Frontend
```

**Flujo de deploy:**
1. `git push` a `main` dispara GitHub Actions
2. Actions hace `docker build` → `docker push` a ECR
3. Actions hace SSH a EC2 → `docker pull` → reinicia container
4. FastAPI descarga `model.pkl` desde S3 al iniciar
5. Credenciales AWS almacenadas en GitHub Secrets

---

## Módulo central: Análisis por zonas

La cancha de cada mapa se divide en una grilla. Las coordenadas X/Y de cada kill se asignan a una celda de la grilla.

**Por cada kill se registra:**
- Jugador
- Mapa
- Zona (celda de la grilla)
- Lado: `ATK` o `DEF`
- Resultado: kill o muerte

**Tabla resultante por jugador:**

```
jugador | mapa   | zona   | lado | kills | muertes | kill_rate
--------|--------|--------|------|-------|---------|----------
TenZ    | Ascent | B site | ATK  |  47   |   12    |   0.80
TenZ    | Ascent | B site | DEF  |  23   |   18    |   0.56
```

**Regla de muestra mínima:** si una combinación zona+lado tiene menos de 10 eventos, no se muestra el stat (o se marca como "muestra insuficiente"). Evita que un 100% con 1 kill engañe el análisis.

**Tamaño de grilla:** 6×4 celdas por mapa (24 zonas). Grillas más pequeñas generan zonas con menos de 10 eventos, lo que activa el umbral de muestra mínima y hace el análisis inútil.

**Normalización de coordenadas:** cada mapa de Valorant tiene rangos de coordenadas distintos. Se normalizan a un espacio [0,1]×[0,1] antes de aplicar la grilla, con un mapeo específico por mapa.

**Mapas en scope inicial (Fase 1):** Ascent, Bind, Haven, Split, Fracture — los 5 con más datos históricos disponibles. El resto se agregan si hay tiempo.

---

## Endpoints de la API

### `POST /predict-match`
Predice probabilidad de victoria para un equipo. **Scope inicial: jugadores con suficiente historial en la API** (mínimo 20 partidas rankeadas registradas). No está diseñado para predecir partidas profesionales ni partidas sin historial.

**Request:**
```json
{
  "team_a": ["TenZ#NA1", "player2#tag", "player3#tag", "player4#tag", "player5#tag"],
  "team_b": ["s1mple#EU1", "player2#tag", "player3#tag", "player4#tag", "player5#tag"],
  "map": "ascent",
  "agents_a": ["Jett", "Sage", "Sova", "Cypher", "Brimstone"],
  "agents_b": ["Reyna", "Killjoy", "Fade", "Viper", "Omen"]
}
```
**Response:** `{ "team_a_win_prob": 0.63, "team_b_win_prob": 0.37, "model": "random_forest_v1", "disclaimer": "Basado en historial individual. No considera sinergia de equipo." }`

---

### `GET /player/{name}/{tag}/card`
Stats generales del jugador.

**Response:**
```json
{
  "name": "TenZ#NA1",
  "acs": 284,
  "kd": 1.42,
  "hs_pct": 31.2,
  "kast": 74.1,
  "best_map": "ascent",
  "best_agent": "Jett",
  "winrate": 0.58
}
```

---

### `GET /player/{name}/{tag}/zones/{map}?side=ATK|DEF|ALL`
Heatmap de kills del jugador en un mapa, filtrado por lado.

**Response:** imagen PNG (matplotlib) con la grilla del mapa coloreada por kill_rate por zona.

---

### `GET /compare/{player_a_name}/{player_a_tag}/{player_b_name}/{player_b_tag}/{map}?side=ATK|DEF|ALL`
Comparativa espacial entre dos jugadores en un mapa.

**Response:** imagen PNG con heatmap split — zonas azules donde domina A, rojas donde domina B, gris donde la diferencia es < 10 puntos porcentuales.

Incluye en el JSON wrapper: `{ "image_url": "...", "disclaimer": "Basado en historial de partidas. No predice el resultado de un duelo individual." }`

---

### `POST /simulate/{name}/{tag}/{map}`
Simulación "¿qué pasa si?" sobre distribución de peleas por zona.

**Request:** distribución deseada de kills por zona (debe sumar 1.0)
**Response:** KDA esperado con la nueva distribución vs KDA histórico real.

---

### `GET /team-zones/{team_a}/{team_b}/{map}`
Dominio histórico por zona entre dos equipos en un mapa.

**Response:** imagen PNG + JSON con zonas de ventaja por equipo.

---

## Modelo de ML (Win/Loss)

- **Target:** Win/Loss binario (Valorant no tiene empates)
- **Algoritmo:** Random Forest Classifier
- **Features iniciales:**
  - ACS promedio del equipo (últimas 20 partidas)
  - K/D promedio
  - Winrate histórico en ese mapa
  - Composición de agentes (encoded)
  - Forma reciente (últimas 5 partidas: W/L streak)
- **Artefacto:** `model.pkl` serializado con joblib → subido a S3 en `models/`
- **Accuracy esperado:** 60-65% (comunicarlo honestamente en el README)
- **Manejo de desbalance:** class_weight='balanced' en Random Forest

---

## Estructura de carpetas

```
valorant-analyzer/
├── data/
│   ├── raw/              # respuestas JSON de la API
│   └── processed/        # CSVs limpios por mapa y jugador
├── notebooks/            # EDA y entrenamiento (Fases 1-2)
├── src/
│   ├── data_loader.py    # fetching desde HenrikDev/Riot API
│   ├── zones.py          # grilla por mapa, asignación de kills a zonas
│   ├── model.py          # entrenamiento y serialización del modelo
│   └── simulation.py     # lógica de simulación what-if
├── api/
│   └── main.py           # FastAPI — endpoints y carga del modelo desde S3
├── frontend/             # HTML/JS simple
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions: build → ECR → EC2
├── Dockerfile
├── requirements.txt
└── README.md             # con GIFs de demo y explicación honesta del modelo
```

---

## Cronograma — 10 semanas

| Semana | Foco | Entregable |
|--------|------|-----------|
| 1–2 | Datos: HenrikDev API, exploración, CSVs en S3 | Notebooks de EDA + datos procesados |
| 3 | Infra AWS: EC2 + ECR + S3 + IAM + GitHub Actions | App dummy corriendo en EC2 con pipeline automático |
| 4 | Modelo Win/Loss + endpoint `/predict-match` | Predictor en producción |
| 5 | Zonas por mapa: grilla, normalización, heatmaps | `/player/zones` funcionando |
| 6 | Player card + ATK/DEF por zona | `/player/card` con filtro de lado |
| 7 | Duelo por zona entre jugadores | `/compare` con heatmap split |
| 8 | Simulación what-if + dominio por equipo | `/simulate` + `/team-zones` |
| 9 | Frontend HTML/JS + integración visual | UI consumiendo todos los endpoints |
| 10 | Polish, README con GIFs, limpieza | Proyecto publicado y listo para CV |

**Regla de corte:** si en semana 7 hay atraso, priorizar semanas 9–10 sobre agregar más features. El deploy y el README son innegociables.

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| HenrikDev API cambia o cae | Migrar a Riot API oficial (dev key ya solicitada en semana 1) |
| Rate limit de Riot API (20 req/s) | Cachear respuestas de partidas ya procesadas en S3 |
| Coordenadas inconsistentes por mapa | Normalizar a [0,1]² por mapa con mapeo explícito |
| Muestra insuficiente por zona | Umbral mínimo de 10 eventos; mostrar aviso en la respuesta |
| IAM de AWS es complejo | Reservar día completo en semana 3; seguir guía paso a paso |
| Quedarse en el ML sin deployar | Pipeline de CI/CD en semana 3, no al final |

---

## Lo que NO hace este sistema

- No predice movimientos en tiempo real
- No reemplaza el análisis de un coach profesional
- No garantiza el resultado de un duelo individual
- El modelo Win/Loss tiene ~60-65% de accuracy — no es una apuesta segura

Estas limitaciones se documentan explícitamente en el README y en los disclaimers de los endpoints. Comunicarlas es señal de madurez técnica, no debilidad.
