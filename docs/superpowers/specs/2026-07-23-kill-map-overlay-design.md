# Design: Overlay descriptivo de kills sobre el mapa real

**Fecha:** 2026-07-23
**Autor:** David Salomón Alvear Luengo
**Estado:** Aprobado

---

## Objetivo

Visualizar las kills reales sobre la **imagen real de cada mapa** de Valorant, en dos
modos alternables:

- **Scatter:** cada kill como un punto en su coordenada exacta (filtrable por jugador,
  lado y resultado).
- **Heatmap:** densidad de ubicaciones de kills (hot spots) sobre el mismo mapa.

Es la **capa descriptiva** que el spec del modelo predictivo
(`2026-07-11-predictive-round-site-model-design.md`) daba por sentada como presentación.
Se construye primero en **Jupyter**; el núcleo queda como funciones puras en `src/` para
reutilizarlo en un port web posterior.

---

## Decisiones clave

| Decisión | Elección | Razón |
|----------|----------|-------|
| Contenido | Scatter + heatmap, alternables | Cubre "colocar las kills" y "un mapa con info sobre las kills" |
| Plataforma | Jupyter (funciones + ipywidgets) primero; web después | Encaja con el flujo actual de Jupyter; el núcleo se reutiliza |
| Imágenes + coords | `valorant-api.com` (oficial) | Trae imagen del minimapa + constantes de transformación para **todo** el pool, sin calibrar a mano |
| Render | matplotlib | Ya es dependencia; sin stack web ni deps pesadas nuevas |
| Interacción | ipywidgets (capa opcional) | Dropdowns + toggle; pero la función se puede llamar directo sin widgets |
| Heatmap | densidad `hexbin` de ubicaciones | Hot spots clásicos; independiente del grid 6×4 de `zones.py` |

---

## Por qué valorant-api y no MAP_BOUNDS

`MAP_BOUNDS` en `zones.py` solo tiene 5 mapas calibrados a mano (el resto son estimados) y
normaliza por min/max lineal — bien para el binning en zonas, pero no para alinear al píxel
de una imagen concreta. `valorant-api.com` entrega, por mapa, la imagen oficial del minimapa
(`displayIcon`) **y** las constantes `xMultiplier / yMultiplier / xScalarToAdd / yScalarToAdd`
que convierten coords del juego a posición normalizada del minimapa — para todo el pool
competitivo, sin API key.

Las coordenadas de kills de HenrikDev (`player_locations_on_kill`, `victim_death_location`)
son coords del mundo del juego, exactamente el input que espera esa transformación. **No se
toca `zones.py` / `MAP_BOUNDS`**: la transformación de render vive aparte, desacoplada del
binning en zonas.

Aviso conocido: hay un **swap de ejes** (X del minimapa usa la `y` del juego) y el
`yMultiplier` es negativo (flip vertical). Se valida con 1–2 kills de posición conocida en la
implementación.

---

## Arquitectura de componentes

**Se reutiliza tal cual:**
- `src/data_loader.py` (`extract_kills`) — fuente de eventos con `map, player, side, result, x, y`.
- `notebooks/explora.ipynb` — donde se añade la celda de visualización (y su fallback de datos demo).

**No se toca:** `src/zones.py`, `MAP_BOUNDS`, la API ni el modelo.

**Se añade:**

### `src/mapviz.py` (nuevo) — núcleo puro (sin matplotlib)
- `get_map_asset(map_name) -> (image_path, transform)`
  Descarga de valorant-api (una vez) la imagen del minimapa y sus 4 constantes; cachea la
  imagen en `data/maps/<map>.png` y las constantes en `data/maps/transforms.json`. Llamadas
  siguientes usan el caché (no re-descargan). Sin red y sin caché → error claro pidiendo
  conexión para la primera vez.
- `game_to_pixel(df, map_name) -> df`
  Devuelve el DataFrame con columnas `px, py` en píxeles de la imagen, aplicando la
  transformación oficial. Función pura. Mapa con multiplicadores en cero (mapas de
  práctica/TDM) → `ValueError`.

### Presentación en el notebook — capa fina
- `plot_kills_on_map(kills, map_name, player=None, side="ALL", mode="scatter")`
  Dibuja la imagen de fondo (`imshow`) y encima **scatter** (puntos) o **heatmap**
  (`hexbin` de densidad) según `mode`. Llamable directo, sin widgets.
- Envoltorio ipywidgets (opcional): dropdowns *mapa / jugador / lado* + toggle
  *scatter ↔ heatmap*. Si no está instalado `ipywidgets`, la función sigue usándose a mano.

---

## Transformación de coordenadas

Normalizado `[0,1]` del `displayIcon`, con swap de ejes:

```
nx = y_game * xMultiplier + xScalarToAdd
ny = x_game * yMultiplier + yScalarToAdd
px = nx * ancho_img
py = ny * alto_img
```

Ejemplo Ascent: `xMultiplier=0.00007`, `yMultiplier=-0.00007`, `xScalarToAdd=0.813895`,
`yScalarToAdd=0.573242`. El `yMultiplier` negativo produce el flip vertical necesario para
que la imagen (origen arriba-izquierda de `imshow`) quede orientada correctamente.

---

## Caché de assets

- `data/maps/<map>.png` — imágenes. **No se versionan** (regenerables, ~1 MB c/u):
  `data/maps/*.png` va a `.gitignore`.
- `data/maps/transforms.json` — constantes por mapa. **Sí se versiona** (diminuto; evita
  depender de la red para las coordenadas).

---

## Flujo de datos

```
kills (extract_kills o datos demo del notebook)
   → filtrar por mapa / jugador / lado
   → game_to_pixel(map_name)      # añade px, py
   → plot_kills_on_map(mode)      # imshow del mapa + scatter | hexbin
```

---

## Manejo de errores

- Mapa sin transformación válida (multiplicadores 0) → `ValueError` explícito.
- Sin kills tras el filtro → se dibuja el mapa vacío con una nota (no rompe).
- Sin red y sin caché en la primera descarga → error claro indicando que se necesita conexión.

---

## Testing

Convenciones del repo (`tests/conftest.py`, mocks con `pytest-mock`, sin red real). Un check
por pieza no trivial:

- `game_to_pixel`: una coord conocida cae en el píxel esperado (atrapa el swap de ejes y el
  signo). Puro, sin red.
- `get_map_asset`: descarga mockeada; la segunda llamada usa caché y **no** re-descarga.
- Filtrado por jugador/lado produce el subconjunto correcto de kills.

---

## Cambios posteriores a la implementación

Registrados aquí para que el spec no quede desalineado con el código:

- **El filtro por `result` se perdió entre este spec y el plan** (el plan omitió el parámetro), y el overlay acabó dibujando muertes bajo un título que decía "kills". Detectado en el review final y corregido: `plot_kills_on_map` acepta `result="ALL"/"kill"/"death"`.
- **Codificación por color y leyenda:** con `result="ALL"` los dos tipos de evento se dibujan en colores distintos (kills azul, muertes naranja) con leyenda, en vez de puntos idénticos con significados opuestos.
- **Barra de color en el heatmap:** el degradado del hexbin representa nº de eventos por celda y no era interpretable sin escala.
- **Encuadre al contenido del minimapa:** los PNG de valorant-api traen ~10% de padding transparente.

## Alcance / No-objetivos (YAGNI)

- **Solo capa descriptiva.** No toca el modelo predictivo ni `zones.py`.
- **Heatmap = densidad `hexbin`.** Pintar el grid 6×4 de `kill_rate` sobre el mapa es
  follow-up (requiere alinear el grid a la transformación oficial).
- **Jupyter primero.** El port web (endpoint JSON + HTML/canvas) es follow-up y reutiliza
  `src/mapviz.py` sin cambios.
- **Sin pan/zoom/hover** (nada de plotly/bokeh) en v1.
- `ipywidgets` es la única dependencia nueva, y solo para la comodidad interactiva; se añade a
  `requirements-dev.txt`.
