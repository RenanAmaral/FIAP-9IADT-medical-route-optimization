from src.baseline import nearest_neighbor_route, random_route
from src.config import DEFAULT_CONFIG
from src.data_loader import get_hospitals, load_points
from src.distance import build_distance_matrix


def test_random_route_visits_all_hospitals():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    hospital_ids = {hospital.idx for hospital in get_hospitals(points)}

    result = random_route(points, distance_matrix, DEFAULT_CONFIG, seed=42)

    assert set(result.chromosome) == hospital_ids
    assert result.is_valid


def test_random_route_is_reproducible_with_same_seed():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    first = random_route(points, distance_matrix, DEFAULT_CONFIG, seed=7)
    second = random_route(points, distance_matrix, DEFAULT_CONFIG, seed=7)

    assert first.chromosome == second.chromosome
    assert first.fitness == second.fitness


def test_nearest_neighbor_visits_all_hospitals():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    hospital_ids = {hospital.idx for hospital in get_hospitals(points)}

    result = nearest_neighbor_route(points, distance_matrix, DEFAULT_CONFIG)

    assert set(result.chromosome) == hospital_ids
    assert result.is_valid


def test_nearest_neighbor_is_deterministic():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    first = nearest_neighbor_route(points, distance_matrix, DEFAULT_CONFIG)
    second = nearest_neighbor_route(points, distance_matrix, DEFAULT_CONFIG)

    assert first.chromosome == second.chromosome


def test_nearest_neighbor_starts_with_hospital_closest_to_origin():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    result = nearest_neighbor_route(points, distance_matrix, DEFAULT_CONFIG)

    hospital_ids = [hospital.idx for hospital in get_hospitals(points)]
    closest_to_origin = min(
        hospital_ids,
        key=lambda hospital_idx: distance_matrix[(0, hospital_idx)],
    )
    assert result.chromosome[0] == closest_to_origin


def test_both_baselines_share_the_same_evaluation():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    random_result = random_route(points, distance_matrix, DEFAULT_CONFIG, seed=1)
    nearest_result = nearest_neighbor_route(points, distance_matrix, DEFAULT_CONFIG)

    for result in (random_result, nearest_result):
        assert result.fitness > 0
        assert result.total_distance > 0
        assert result.decoded_route[0] == 0
        assert result.decoded_route[-1] == 0
