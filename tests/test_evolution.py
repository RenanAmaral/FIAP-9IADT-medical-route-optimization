from src.baseline import nearest_neighbor_route
from src.data_loader import get_hospitals, load_points
from src.distance import build_distance_matrix
from src.genetic_algorithm import run_genetic_algorithm
from src.models import Config


def make_config(**overrides):
    base = dict(
        vehicle_capacity=100,
        lambda_priority=5.0,
        lambda_supply=10.0,
        pop_size=20,
        generations=15,
        mutation_rate=0.2,
        crossover_rate=0.9,
        tournament_size=3,
        n_elite=2,
        seed=42,
    )
    base.update(overrides)
    return Config(**base)


def test_run_returns_valid_best_individual():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    hospital_ids = {hospital.idx for hospital in get_hospitals(points)}

    result = run_genetic_algorithm(points, distance_matrix, make_config())

    assert result.best.is_valid
    assert set(result.best.chromosome) == hospital_ids
    assert result.best.decoded_route[0] == 0
    assert result.best.decoded_route[-1] == 0


def test_history_has_one_record_per_generation():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = make_config(generations=15)

    result = run_genetic_algorithm(points, distance_matrix, config)

    assert len(result.history) == 15
    assert [record.generation for record in result.history] == list(range(15))


def test_best_fitness_never_worse_than_first_generation():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    result = run_genetic_algorithm(points, distance_matrix, make_config())

    assert result.best.fitness <= result.history[0].best_fitness


def test_best_fitness_is_non_increasing_across_generations():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    result = run_genetic_algorithm(points, distance_matrix, make_config())

    best_per_generation = [record.best_fitness for record in result.history]
    assert best_per_generation == sorted(best_per_generation, reverse=True)


def test_run_is_reproducible_with_same_seed():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)

    first = run_genetic_algorithm(points, distance_matrix, make_config(seed=7))
    second = run_genetic_algorithm(points, distance_matrix, make_config(seed=7))

    assert first.best.chromosome == second.best.chromosome
    assert first.best.fitness == second.best.fitness


def test_ga_is_at_least_as_good_as_nearest_neighbor():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = make_config()

    nearest = nearest_neighbor_route(points, distance_matrix, config)
    result = run_genetic_algorithm(points, distance_matrix, config)

    assert result.best.fitness <= nearest.fitness
