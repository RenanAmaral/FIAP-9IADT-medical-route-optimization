import pytest

from src.data_loader import load_points
from src.distance import build_distance_matrix, get_distance, haversine_distance


def test_haversine_distance_returns_positive_distance_for_distinct_points():
    distance = haversine_distance(-23.5505, -46.6333, -23.5600, -46.6400)

    assert distance > 0


def test_haversine_distance_same_point_is_zero():
    distance = haversine_distance(-23.5505, -46.6333, -23.5505, -46.6333)

    assert distance == pytest.approx(0.0)


def test_haversine_distance_matches_known_value_in_km():
    # Hospital Central -> Unidade Hospitalar A: ~1.26 km
    distance = haversine_distance(-23.5505, -46.6333, -23.5600, -46.6400)

    assert distance == pytest.approx(1.26, abs=0.05)


def test_distance_matrix_contains_expected_pairs():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    assert (0, 1) in distance_matrix
    assert (1, 0) in distance_matrix
    assert get_distance(0, 0, distance_matrix) == pytest.approx(0.0)
