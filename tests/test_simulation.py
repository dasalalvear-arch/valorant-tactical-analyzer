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
