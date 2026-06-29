from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.fitness import calculate_priority_penalty, evaluate
from src.models import Config


def test_fitness_returns_numeric_value():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    result = evaluate([3, 1, 5, 2], points, distance_matrix, DEFAULT_CONFIG)

    assert isinstance(result.fitness, float)
    assert result.fitness > 0


def test_high_priority_earlier_has_lower_priority_penalty():
    points = load_points("data/pontos_entrega.csv")
    points_by_idx = {point.idx: point for point in points}

    early_high_priority = calculate_priority_penalty([1, 3], points_by_idx)
    late_high_priority = calculate_priority_penalty([3, 1], points_by_idx)

    assert early_high_priority < late_high_priority


def test_more_resupplies_has_greater_supply_penalty():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    low_capacity = Config(vehicle_capacity=50, lambda_priority=5.0, lambda_supply=10.0)

    fewer_resupplies = evaluate([7], points, distance_matrix, low_capacity)
    more_resupplies = evaluate([7, 10], points, distance_matrix, low_capacity)

    assert more_resupplies.supply_penalty > fewer_resupplies.supply_penalty


def test_evaluation_result_contains_expected_fields():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    result = evaluate([3, 1, 5, 2], points, distance_matrix, DEFAULT_CONFIG)

    assert result.decoded_route
    assert result.fitness > 0
    assert result.total_distance > 0
    assert result.priority_penalty > 0
    assert result.supply_penalty >= 0
    assert result.resupply_count >= 0
