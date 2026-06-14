"""
Simulación what-if sobre las estadísticas por zona (no entrena nada).

Responde: "¿cómo cambiaría el kill_rate de un jugador si redistribuyera
sus peleas entre otras zonas del mapa?". Trabaja sobre los zone_stats ya
calculados por zones.compute_zone_stats.
"""
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
    # La distribución debe sumar 1.0 (con tolerancia): son fracciones de peleas.
    total_weight = sum(new_distribution.values())
    if abs(total_weight - 1.0) > tolerance:
        raise ValueError(f"distribución debe sumar 1.0, recibió {total_weight:.2f}")

    # Stats del jugador para ese lado y mapa, descartando zonas con muestra insuficiente.
    player_stats = zone_stats[
        (zone_stats["player"] == player)
        & (zone_stats["map"] == map_name)
        & (zone_stats["side"] == side)
        & (~zone_stats["insufficient_sample"])
    ]

    # Línea base: kill_rate histórico real del jugador (None si no hay datos).
    total_events = player_stats["total_events"].sum()
    historical_kill_rate = (
        player_stats["kills"].sum() / total_events
        if not player_stats.empty and total_events > 0
        else None
    )

    # Promedio ponderado: por cada zona destino, su kill_rate por el peso asignado.
    simulated_kill_rate = 0.0
    for (zone_row, zone_col), weight in new_distribution.items():
        zone = player_stats[
            (player_stats["zone_row"] == zone_row)
            & (player_stats["zone_col"] == zone_col)
        ]
        if zone.empty:
            # Zona sin datos: caemos al promedio histórico (0.5 si no hay nada).
            # Usamos `is None` y no `or` para no confundir un kill_rate real de 0.0.
            kr = historical_kill_rate if historical_kill_rate is not None else 0.5
        else:
            kr = float(zone.iloc[0]["kill_rate"])
        simulated_kill_rate += weight * kr

    baseline = historical_kill_rate if historical_kill_rate is not None else 0.0
    return {
        "player": player,
        "map": map_name,
        "side": side,
        "historical_kill_rate": (
            round(historical_kill_rate, 4) if historical_kill_rate is not None else None
        ),
        "simulated_kill_rate": round(simulated_kill_rate, 4),
        # delta > 0 → la redistribución mejora; < 0 → empeora.
        "delta": round(simulated_kill_rate - baseline, 4),
        "disclaimer": "Simulación basada en historial. No predice resultados reales.",
    }
