# Design: Modelo predictivo de ronda y site (dominio espacial)

**Fecha:** 2026-07-11
**Autor:** David Salomón Alvear Luengo
**Estado:** Aprobado

---

## Objetivo

Construir un modelo **predictivo** que, *antes de una ronda*, estime qué lado está
favorecido a partir del **dominio espacial histórico** de los jugadores, y aplicarlo a
un site concreto para responder "¿quién está favorecido para ganar el site B?".

Dos fases, ambas en v1:

- **Fase 1 (motor ML):** predecir `P(gana el ataque la ronda)` a partir de perfiles
  espaciales históricos, con split temporal. Aquí está el rigor de CV.
- **Fase 2 (gancho visual):** aplicar ese mismo modelo a las zonas de un site → equipo
  favorito + probabilidad para ese site, dibujado sobre el mapa.

La capa **descriptiva** existente (stats por zona sobre el mapa) sirve de presentación
para ambas.

---

## Por qué este enfoque y no el ingenuo

La idea intuitiva "más kills en la zona → gana el site" tiene tres riesgos que este
diseño evita explícitamente:

1. **Fuga de datos / circularidad.** Matar *es* el mecanismo de ganar la ronda. Usar los
   kills de una ronda para predecir su propio resultado da precisión alta y valor cero
   (*target leakage*). **Se evita** prediciendo con el pasado del jugador, nunca con la
   ronda misma (ver ventanas temporales).
2. **Muestra pequeña.** Trocear por jugador × mapa × zona × lado deja celdas con pocas
   muestras. **Se mitiga** con la opción Pool (historial completo de todos los jugadores)
   y respetando el flag `insufficient_sample` de `zones.py` con fallback.
3. **Puede no batir un baseline tonto.** Ganar una ronda depende de mucho más que las
   kills de un jugador. **Se afronta** midiendo siempre contra baselines; si no los bate,
   es un resultado honesto que se reporta, no un fracaso.

---

## Decisiones clave

| Decisión | Elección | Razón |
|----------|----------|-------|
| Etiqueta | Ganador de la ronda (ATK vs DEF) | Muchos datos, etiqueta limpia y siempre presente |
| Momento de uso | Antes de la ronda (solo historial previo) | Sin leakage; predicción útil de verdad |
| De quién hay datos | **Pool** de jugadores fijos | Historial completo de los 10 → señal espacial limpia |
| Modelo | `LogisticRegression` principal | Coeficientes por zona = interpretabilidad = la tesis en números |
| Site (Fase 2) | Aplicación del mismo modelo | No es un modelo aparte; reutiliza el motor |
| Visualización | Mapa HTML interactivo, **un solo mapa** | Máximo "vistoso" por esfuerzo; vanilla JS, sin framework |

---

## Arquitectura de componentes

**Se reutilizan tal cual:**

- `src/data_loader.py` — baja partidas + aplana kills/deaths.
- `src/zones.py` — `assign_zones` + `compute_zone_stats`. **Fuente de los perfiles
  espaciales por jugador** (`kill_rate` por jugador/mapa/zona/lado). El corazón ya existe.
- `src/simulation.py` — what-if. Se queda; no es central a la predicción.

**Se añade / modifica:**

1. **`src/data_loader.py` → nueva `extract_rounds(matches)`.** DataFrame de rondas:
   `match_id, map, round_idx, fecha, atk_won (bool), plant_site (A/B/C/None)` + rosters de
   cada equipo. Aporta la **etiqueta** y quién jugó. Es la pieza que hoy falta.
2. **`src/zones.py` → mapa zona→site.** Dict estático por mapa: qué celdas de la grid 6×4
   son *site A*, *site B* o *mid* (de los callouts del mapa). Habilita "enfocar" el modelo
   en un site.
3. **`src/features.py` (nuevo) — el anti-leakage vive aquí.** Construye el vector de
   features por ronda usando solo partidas anteriores (ver abajo).
4. **`src/model.py` → se reescribe.** Fuera el modelo circular de winrate. Entra el modelo
   honesto de ronda con split temporal y comparación contra baselines.
5. **Predicción de site (Fase 2)** — función (en `model.py` o `src/predict.py`) que, dado
   un matchup + site, reutiliza el modelo pesando las features a las zonas de ese site.
6. **`api/main.py` (nuevo) — endpoints:** stats por zona (descriptivo), predecir ronda
   (rosters), predecir site (rosters + site).
7. **Visualización** — página HTML estática servida por FastAPI (ver sección).

---

## Datos y ventanas temporales (clave anti-leakage)

Las partidas del Pool se parten por fecha en dos ventanas:

- **Ventana de historial** (las más viejas): se construye el perfil espacial de cada
  jugador **una sola vez** aquí.
- **Ventana de predicción** (las más nuevas): cada ronda de aquí es una muestra a predecir.

Como el perfil sale solo del historial y la ronda a predecir es posterior, **los kills de
una ronda nunca entran en sus propias features**. Ese es todo el mecanismo, y es limpio.

Simplificación deliberada: perfiles "congelados" en el corte historial/predicción en vez
de recalcular el corte ronda a ronda. Suficiente para el Pool; perfiles deslizantes por
fecha serían un cambio localizado en `features.py` si se quisiera después.

---

## Features

Vector de features de una ronda:

- **Perfil de un jugador** = su `kill_rate` en las 24 zonas (6×4), filtrado al lado que
  juega esa ronda (ATK/DEF — reutiliza la dimensión de lado de `zone_stats`). Zonas con
  `insufficient_sample` → fallback al `kill_rate` global del jugador en el mapa (misma
  lógica de fallback que `simulation.py`).
- Se promedian los 5 perfiles de cada equipo → `perfil_atk[24]`, `perfil_def[24]`.
- **Feature final = `perfil_atk − perfil_def`** → 24 valores: cuánto domina el ataque cada
  zona respecto a la defensa.
- **Etiqueta** = `atk_won` (1/0).

---

## Modelo y evaluación

**Modelo principal: `LogisticRegression`.** Con 24 features, los coeficientes indican qué
zonas predicen ganar la ronda — interpretabilidad que *es* la tesis del proyecto. Se deja
`RandomForest` como comparación de una línea, pero no es el protagonista.

**Evaluación (la honestidad del proyecto):**

- Split temporal train/test dentro de la ventana de predicción.
- Métricas: **AUC + accuracy** sobre el test temporal.
- **Baselines a batir:** (a) clase mayoritaria; (b) "gana el equipo con mejor winrate
  global". Batirlos → la señal espacial es real. No batirlos → resultado honesto que se
  reporta (sigue demostrando rigor).

---

## Predicción de site (Fase 2)

Reutiliza el mismo modelo logístico. Para "¿quién gana el site B?" se arma el vector con el
diferencial real solo en las zonas de site B (del mapa zona→site) y neutro (0) en el resto
→ el modelo devuelve equipo favorito + probabilidad **para ese site**. Es una aproximación
consistente con el motor, no un modelo aparte.

---

## Visualización

**Mapa HTML interactivo, acotado a un solo mapa** (el que ya tiene `MAP_BOUNDS` calibrado,
Bind): página estática con un minimapa de fondo, las 24 zonas dibujadas como grid encima,
`fetch` al JSON de la API, coloreado de zonas por dominio + favorito del site + tooltips con
stats por zona. Vanilla JS, sin framework; FastAPI sirve el HTML.

Red de seguridad: si la calibración del overlay se complica, un PNG de matplotlib (heatmap
de zonas + site anotado) cubre lo visual sin quedarse sin nada.

---

## Testing

Se siguen las convenciones del repo (`tests/conftest.py`, mocks de API con `pytest-mock`,
sin red real). Un check por pieza no trivial:

- `extract_rounds` parsea `atk_won`, `plant_site` y rosters desde un match mockeado.
- `features.py`: vector de tamaño 24; **el corte temporal se respeta** (una ronda no
  arrastra sus propios kills); agregación por equipo y fallback correctos.
- `model.py`: sobre un dataset sintético *separable* (equipo que domina las zonas gana
  siempre), el modelo lo aprende y **le gana al baseline** — sanity check.
- Predicción de site: perfil sintético que favorece a un equipo en zonas de site B →
  devuelve ese equipo como favorito.

---

## Alcance / No-objetivos (YAGNI)

- **Un solo mapa** con visualización interactiva en v1 (Bind). El resto, después.
- **No** modelo aparte para site; es aplicación del de ronda.
- **No** perfiles deslizantes por fecha en v1; perfiles congelados.
- **No** overtime bien resuelto (hereda la aproximación de lado de `_round_attackers`).
- **No** React ni front-end con framework.

---

## Riesgos conocidos

- **Cold-start fuera del Pool:** el demo solo predice para jugadores del Pool. Aceptable en
  portfolio; documentado como tal.
- **El modelo podría no batir el baseline:** mitigado convirtiéndolo en hallazgo reportado,
  no en fracaso.
- **`plant_site` solo en rondas con plant:** la Fase 2 depende del mapa zona→site, no del
  `plant_site` de cada ronda, así que no bloquea; `plant_site` es solo para etiquetar/analizar.
