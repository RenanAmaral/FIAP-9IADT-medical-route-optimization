import random
from typing import List, Optional

from src.data_loader import get_hospitals, get_origin
from src.distance import DistanceMatrix, get_distance
from src.fitness import evaluate
from src.models import Config, EvaluationResult, Point


def random_route(
    points: List[Point],
    distance_matrix: DistanceMatrix,
    config: Config,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """Referencia fraca: embaralha os hospitais em ordem aleatoria."""
    rng = random.Random(seed)
    hospital_ids = [hospital.idx for hospital in get_hospitals(points)]
    rng.shuffle(hospital_ids)
    return evaluate(hospital_ids, points, distance_matrix, config)


def nearest_neighbor_route(
    points: List[Point],
    distance_matrix: DistanceMatrix,
    config: Config,
) -> EvaluationResult:
    """Heuristica gulosa: a cada passo escolhe o hospital nao visitado mais proximo."""
    origin = get_origin(points)
    remaining = [hospital.idx for hospital in get_hospitals(points)]
    chromosome: List[int] = []
    current_idx = origin.idx

    while remaining:
        nearest_idx = min(
            remaining,
            key=lambda hospital_idx: get_distance(current_idx, hospital_idx, distance_matrix),
        )
        chromosome.append(nearest_idx)
        remaining.remove(nearest_idx)
        current_idx = nearest_idx

    return evaluate(chromosome, points, distance_matrix, config)
