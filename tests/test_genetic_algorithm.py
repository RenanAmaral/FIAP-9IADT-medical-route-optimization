import random

from src.config import DEFAULT_CONFIG
from src.data_loader import get_hospitals, load_points
from src.distance import build_distance_matrix
from src.genetic_algorithm import (
    initial_population,
    inversion_mutation,
    mutate,
    order_crossover,
    random_chromosome,
    select_elite,
    swap_mutation,
    tournament_selection,
)
from src.models import Config


HOSPITAL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def is_permutation_of(chromosome, reference):
    return sorted(chromosome) == sorted(reference)


def test_random_chromosome_is_a_permutation():
    rng = random.Random(0)

    chromosome = random_chromosome(HOSPITAL_IDS, rng)

    assert is_permutation_of(chromosome, HOSPITAL_IDS)


def test_random_chromosome_is_reproducible_with_seeded_rng():
    first = random_chromosome(HOSPITAL_IDS, random.Random(123))
    second = random_chromosome(HOSPITAL_IDS, random.Random(123))

    assert first == second


def test_initial_population_has_pop_size_valid_individuals():
    points = load_points("data/pontos_entrega.csv")
    config = Config(vehicle_capacity=100, lambda_priority=5.0, lambda_supply=10.0, pop_size=30)
    hospital_ids = [hospital.idx for hospital in get_hospitals(points)]

    population = initial_population(points, config, random.Random(1))

    assert len(population) == 30
    assert all(is_permutation_of(individual, hospital_ids) for individual in population)


def test_initial_population_can_seed_nearest_neighbor():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = Config(vehicle_capacity=100, lambda_priority=5.0, lambda_supply=10.0, pop_size=10)

    population = initial_population(
        points,
        config,
        random.Random(1),
        distance_matrix=distance_matrix,
        seed_nearest_neighbor=True,
    )

    assert len(population) == 10
    assert population[0][0] == 1  # hospital mais proximo da origem


def test_tournament_selection_returns_best_when_everyone_competes():
    population = [[1, 2, 3], [3, 2, 1], [2, 1, 3]]
    fitnesses = [50.0, 10.0, 30.0]

    winner = tournament_selection(population, fitnesses, tournament_size=3, rng=random.Random(0))

    assert winner == [3, 2, 1]


def test_tournament_selection_returns_a_copy():
    population = [[1, 2, 3]]
    winner = tournament_selection(population, [10.0], tournament_size=1, rng=random.Random(0))

    winner.append(99)
    assert population[0] == [1, 2, 3]


def test_order_crossover_keeps_permutation_invariant():
    parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
    parent2 = [8, 7, 6, 5, 4, 3, 2, 1]
    rng = random.Random(5)

    for _ in range(50):
        child = order_crossover(parent1, parent2, rng)
        assert is_permutation_of(child, parent1)


def test_order_crossover_with_identical_parents_returns_same_permutation():
    parent = [1, 2, 3, 4, 5]

    child = order_crossover(parent, parent, random.Random(3))

    assert child == parent


def test_swap_mutation_keeps_permutation_and_changes_at_most_two_positions():
    chromosome = [1, 2, 3, 4, 5]

    mutated = swap_mutation(chromosome, random.Random(2))

    assert is_permutation_of(mutated, chromosome)
    differences = sum(1 for a, b in zip(chromosome, mutated) if a != b)
    assert differences in (0, 2)


def test_inversion_mutation_keeps_permutation_invariant():
    chromosome = [1, 2, 3, 4, 5, 6]

    for seed in range(20):
        mutated = inversion_mutation(chromosome, random.Random(seed))
        assert is_permutation_of(mutated, chromosome)


def test_mutate_with_zero_rate_does_not_change_chromosome():
    chromosome = [1, 2, 3, 4, 5]
    config = Config(vehicle_capacity=100, lambda_priority=5.0, lambda_supply=10.0, mutation_rate=0.0)

    mutated = mutate(chromosome, config, random.Random(0))

    assert mutated == chromosome


def test_mutate_with_full_rate_keeps_permutation_invariant():
    chromosome = [1, 2, 3, 4, 5]
    config = Config(vehicle_capacity=100, lambda_priority=5.0, lambda_supply=10.0, mutation_rate=1.0)

    for seed in range(20):
        mutated = mutate(chromosome, config, random.Random(seed))
        assert is_permutation_of(mutated, chromosome)


def test_select_elite_returns_best_individuals_sorted():
    population = [[1, 2], [3, 4], [5, 6]]
    fitnesses = [30.0, 10.0, 20.0]

    elite = select_elite(population, fitnesses, n_elite=2)

    assert elite == [[3, 4], [5, 6]]


def test_select_elite_returns_copies():
    population = [[1, 2, 3]]
    elite = select_elite(population, [10.0], n_elite=1)

    elite[0].append(99)
    assert population[0] == [1, 2, 3]
