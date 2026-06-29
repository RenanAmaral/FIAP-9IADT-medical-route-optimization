from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.decoder import decode_route
from src.distance import build_distance_matrix
from src.models import Config


def test_decoder_adds_origin_at_start_and_end():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    decoded = decode_route([1, 2, 3], points, distance_matrix, DEFAULT_CONFIG)

    assert decoded.route[0] == 0
    assert decoded.route[-1] == 0


def test_decoder_visits_all_hospitals_from_chromosome():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    chromosome = [3, 1, 5, 2]

    decoded = decode_route(chromosome, points, distance_matrix, DEFAULT_CONFIG)

    assert [idx for idx in decoded.route if idx in chromosome] == chromosome


def test_decoder_inserts_supply_when_load_is_not_enough():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = Config(vehicle_capacity=50, lambda_priority=5.0, lambda_supply=10.0)

    decoded = decode_route([7, 10], points, distance_matrix, config)

    assert decoded.resupply_count == 1
    assert any(idx >= 100 for idx in decoded.route)


def test_decoder_restores_load_after_supply():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = Config(vehicle_capacity=50, lambda_priority=5.0, lambda_supply=10.0)

    decoded = decode_route([7, 10], points, distance_matrix, config)

    assert decoded.remaining_load == 20


def test_decoder_does_not_insert_supply_when_load_is_enough():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    decoded = decode_route([3, 12], points, distance_matrix, DEFAULT_CONFIG)

    assert decoded.resupply_count == 0
    assert all(idx < 100 for idx in decoded.route)


def test_decoder_marks_invalid_when_demand_exceeds_capacity():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = Config(vehicle_capacity=20, lambda_priority=5.0, lambda_supply=10.0)

    decoded = decode_route([1], points, distance_matrix, config)

    assert not decoded.is_valid
    assert decoded.errors
