import random
from math import inf
from typing import List, Optional

from src.baseline import nearest_neighbor_route
from src.data_loader import get_hospitals
from src.distance import DistanceMatrix
from src.fitness import evaluate
from src.models import (
    Config,
    EvaluationResult,
    EvolutionResult,
    GenerationRecord,
    Point,
)


def random_chromosome(hospital_ids: List[int], rng: random.Random) -> List[int]:
    """Permutacao aleatoria dos hospitais (sem origem, sem abastecimento, sem repeticao)."""
    chromosome = list(hospital_ids)
    rng.shuffle(chromosome)
    return chromosome


def initial_population(
    points: List[Point],
    config: Config,
    rng: random.Random,
    distance_matrix: Optional[DistanceMatrix] = None,
    seed_nearest_neighbor: bool = False,
) -> List[List[int]]:
    """pop_size permutacoes; opcionalmente semeia 1 individuo com a rota nearest neighbor."""
    hospital_ids = [hospital.idx for hospital in get_hospitals(points)]
    population: List[List[int]] = []

    if seed_nearest_neighbor and distance_matrix is not None:
        nn_result = nearest_neighbor_route(points, distance_matrix, config)
        population.append(list(nn_result.chromosome))

    while len(population) < config.pop_size:
        population.append(random_chromosome(hospital_ids, rng))

    return population


def tournament_selection(
    population: List[List[int]],
    fitnesses: List[float],
    tournament_size: int,
    rng: random.Random,
) -> List[int]:
    """Sorteia tournament_size competidores e devolve uma copia do de menor fitness."""
    competitors = rng.sample(range(len(population)), min(tournament_size, len(population)))
    winner = min(competitors, key=lambda index: fitnesses[index])
    return list(population[winner])


def order_crossover(
    parent1: List[int],
    parent2: List[int],
    rng: random.Random,
) -> List[int]:
    """Order Crossover (OX): preserva um segmento de parent1 e completa com parent2 em ordem."""
    size = len(parent1)
    if size < 2:
        return list(parent1)

    start, end = sorted(rng.sample(range(size), 2))
    child: List[Optional[int]] = [None] * size
    child[start : end + 1] = parent1[start : end + 1]
    segment = set(parent1[start : end + 1])

    parent2_sequence = parent2[end + 1 :] + parent2[: end + 1]
    fill_genes = [gene for gene in parent2_sequence if gene not in segment]
    empty_positions = [
        position
        for position in list(range(end + 1, size)) + list(range(0, end + 1))
        if child[position] is None
    ]

    for position, gene in zip(empty_positions, fill_genes):
        child[position] = gene

    return [gene for gene in child if gene is not None]


def swap_mutation(chromosome: List[int], rng: random.Random) -> List[int]:
    """Troca dois genes de posicao."""
    mutated = list(chromosome)
    if len(mutated) < 2:
        return mutated
    first, second = rng.sample(range(len(mutated)), 2)
    mutated[first], mutated[second] = mutated[second], mutated[first]
    return mutated


def inversion_mutation(chromosome: List[int], rng: random.Random) -> List[int]:
    """Inverte a ordem de um trecho do cromossomo."""
    mutated = list(chromosome)
    if len(mutated) < 2:
        return mutated
    start, end = sorted(rng.sample(range(len(mutated)), 2))
    mutated[start : end + 1] = reversed(mutated[start : end + 1])
    return mutated


def mutate(chromosome: List[int], config: Config, rng: random.Random) -> List[int]:
    """Aplica swap ou inversion com probabilidade mutation_rate; senao devolve uma copia."""
    if rng.random() >= config.mutation_rate:
        return list(chromosome)
    operator = rng.choice((swap_mutation, inversion_mutation))
    return operator(chromosome, rng)


def select_elite(
    population: List[List[int]],
    fitnesses: List[float],
    n_elite: int,
) -> List[List[int]]:
    """Devolve copias dos n_elite individuos de menor fitness."""
    ranked = sorted(zip(fitnesses, population), key=lambda pair: pair[0])
    return [list(chromosome) for _, chromosome in ranked[:n_elite]]


def _average_fitness(fitnesses: List[float]) -> float:
    finite = [fitness for fitness in fitnesses if fitness != inf]
    return sum(finite) / len(finite) if finite else inf


def _breed_offspring(
    population: List[List[int]],
    fitnesses: List[float],
    config: Config,
    rng: random.Random,
) -> List[int]:
    parent1 = tournament_selection(population, fitnesses, config.tournament_size, rng)
    parent2 = tournament_selection(population, fitnesses, config.tournament_size, rng)
    if rng.random() < config.crossover_rate:
        child = order_crossover(parent1, parent2, rng)
    else:
        child = list(parent1)
    return mutate(child, config, rng)


def run_genetic_algorithm(
    points: List[Point],
    distance_matrix: DistanceMatrix,
    config: Config,
    rng: Optional[random.Random] = None,
    seed_nearest_neighbor: bool = True,
) -> EvolutionResult:
    """Laco evolutivo: avaliar -> registrar -> selecionar -> cruzar -> mutar -> elitismo."""
    rng = rng or random.Random(config.seed)
    population = initial_population(
        points,
        config,
        rng,
        distance_matrix=distance_matrix,
        seed_nearest_neighbor=seed_nearest_neighbor,
    )

    history: List[GenerationRecord] = []
    best: Optional[EvaluationResult] = None

    for generation in range(config.generations):
        results = [evaluate(chromosome, points, distance_matrix, config) for chromosome in population]
        fitnesses = [result.fitness for result in results]

        generation_best = min(results, key=lambda result: result.fitness)
        if best is None or generation_best.fitness < best.fitness:
            best = generation_best

        history.append(
            GenerationRecord(
                generation=generation,
                best_fitness=generation_best.fitness,
                best_distance=generation_best.total_distance,
                average_fitness=_average_fitness(fitnesses),
            )
        )

        next_population = select_elite(population, fitnesses, config.n_elite)
        while len(next_population) < config.pop_size:
            next_population.append(_breed_offspring(population, fitnesses, config, rng))
        population = next_population

    return EvolutionResult(best=best, history=history, generations=config.generations)
