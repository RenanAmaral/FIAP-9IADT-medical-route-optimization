import time
from dataclasses import dataclass, replace
from typing import List, Optional

from src.baseline import nearest_neighbor_route, random_route
from src.distance import DistanceMatrix
from src.genetic_algorithm import run_genetic_algorithm
from src.models import Config, EvaluationResult, EvolutionResult, Point


CRITICAL_PRIORITY = "ALTA"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    pop_size: int
    generations: int
    mutation_rate: float


# E1 (50/100/5%), E2 (100/200/10%), E3 (200/300/15%)
DEFAULT_EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("E1", 50, 100, 0.05),
    ExperimentSpec("E2", 100, 200, 0.10),
    ExperimentSpec("E3", 200, 300, 0.15),
]


@dataclass
class MethodMetrics:
    method: str
    distance: float
    fitness: float
    resupply_count: int
    avg_critical_position: float
    runtime_seconds: float


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    random: MethodMetrics
    nearest_neighbor: MethodMetrics
    genetic: MethodMetrics
    evolution: EvolutionResult
    gain_vs_random_pct: float
    gain_vs_nearest_pct: float


def average_critical_position(chromosome: List[int], points: List[Point]) -> float:
    """Posicao media de visita (1-indexada) dos hospitais ALTA. Menor = atendidos mais cedo."""
    priority_by_idx = {point.idx: point.priority for point in points}
    positions = [
        position
        for position, hospital_idx in enumerate(chromosome, start=1)
        if priority_by_idx[hospital_idx] == CRITICAL_PRIORITY
    ]
    return sum(positions) / len(positions) if positions else 0.0


def _metrics(method: str, result: EvaluationResult, points: List[Point], runtime: float) -> MethodMetrics:
    return MethodMetrics(
        method=method,
        distance=result.total_distance,
        fitness=result.fitness,
        resupply_count=result.resupply_count,
        avg_critical_position=average_critical_position(result.chromosome, points),
        runtime_seconds=runtime,
    )


def run_experiment(
    spec: ExperimentSpec,
    points: List[Point],
    distance_matrix: DistanceMatrix,
    base_config: Config,
    seed: int = 42,
) -> ExperimentResult:
    """Roda aleatoria, nearest neighbor e GA para uma configuracao, com seed fixa."""
    config = replace(
        base_config,
        pop_size=spec.pop_size,
        generations=spec.generations,
        mutation_rate=spec.mutation_rate,
        seed=seed,
    )

    start = time.perf_counter()
    random_result = random_route(points, distance_matrix, config, seed=seed)
    random_time = time.perf_counter() - start

    start = time.perf_counter()
    nearest_result = nearest_neighbor_route(points, distance_matrix, config)
    nearest_time = time.perf_counter() - start

    start = time.perf_counter()
    evolution = run_genetic_algorithm(points, distance_matrix, config)
    genetic_time = time.perf_counter() - start
    genetic_result = evolution.best

    gain_vs_random = 100.0 * (random_result.fitness - genetic_result.fitness) / random_result.fitness
    gain_vs_nearest = 100.0 * (nearest_result.fitness - genetic_result.fitness) / nearest_result.fitness

    return ExperimentResult(
        spec=spec,
        random=_metrics("Aleatoria", random_result, points, random_time),
        nearest_neighbor=_metrics("Nearest neighbor", nearest_result, points, nearest_time),
        genetic=_metrics("Algoritmo genetico", genetic_result, points, genetic_time),
        evolution=evolution,
        gain_vs_random_pct=gain_vs_random,
        gain_vs_nearest_pct=gain_vs_nearest,
    )


def run_all_experiments(
    points: List[Point],
    distance_matrix: DistanceMatrix,
    base_config: Config,
    specs: Optional[List[ExperimentSpec]] = None,
    seed: int = 42,
) -> List[ExperimentResult]:
    specs = specs if specs is not None else DEFAULT_EXPERIMENTS
    return [run_experiment(spec, points, distance_matrix, base_config, seed) for spec in specs]


def format_comparison_table(results: List[ExperimentResult]) -> str:
    """Tabela comparativa aleatoria x nearest neighbor x GA por experimento."""
    header = (
        f"{'Exp':<4}{'Metodo':<20}{'Dist(km)':>10}{'Reab':>6}"
        f"{'PosCrit':>9}{'Fitness':>10}{'Tempo(s)':>10}"
    )
    lines = [header, "-" * len(header)]

    for result in results:
        spec = result.spec
        title = f"{spec.name} (pop={spec.pop_size}, ger={spec.generations}, mut={spec.mutation_rate:.0%})"
        lines.append(title)
        for metrics in (result.random, result.nearest_neighbor, result.genetic):
            lines.append(
                f"{'':<4}{metrics.method:<20}{metrics.distance:>10.2f}{metrics.resupply_count:>6}"
                f"{metrics.avg_critical_position:>9.2f}{metrics.fitness:>10.2f}{metrics.runtime_seconds:>10.3f}"
            )
        lines.append(
            f"{'':<4}Ganho GA vs aleatoria: {result.gain_vs_random_pct:5.1f}%  |  "
            f"vs nearest neighbor: {result.gain_vs_nearest_pct:5.1f}%"
        )
        lines.append("")

    return "\n".join(lines).rstrip()
