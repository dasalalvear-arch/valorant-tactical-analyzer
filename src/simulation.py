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
        (zone_stats["player"] == player)
        & (zone_stats["map"] == map_name)
        & (zone_stats["side"] == side)
        & (~zone_stats["insufficient_sample"])
    ]

    total_events = player_stats["total_events"].sum()
    historical_kill_rate = (
        player_stats["kills"].sum() / total_events
        if not player_stats.empty and total_events > 0
        else None
    )

    simulated_kill_rate = 0.0
    for (zone_row, zone_col), weight in new_distribution.items():
        zone = player_stats[
            (player_stats["zone_row"] == zone_row)
            & (player_stats["zone_col"] == zone_col)
        ]
        if zone.empty:
            # Zona sin datos: caer al promedio histórico del jugador (0.5 si no hay nada).
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
        "delta": round(simulated_kill_rate - baseline, 4),
        "disclaimer": "Simulación basada en historial. No predice resultados reales.",
    }
