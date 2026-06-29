from typing import Dict, List

from src.decoder import decode_route, validate_chromosome
from src.distance import DistanceMatrix
from src.models import Config, EvaluationResult, Point


PRIORITY_WEIGHTS: Dict[str, int] = {
    "ALTA": 3,
    "MEDIA": 2,
    "BAIXA": 1,
}


def calculate_priority_penalty(chromosome: List[int], points_by_idx: Dict[int, Point]) -> float:
    penalty = 0.0
    for position, hospital_idx in enumerate(chromosome, start=1):
        point = points_by_idx[hospital_idx]
        priority_weight = PRIORITY_WEIGHTS[point.priority]
        penalty += priority_weight * position
    return penalty


def calculate_supply_penalty(resupply_count: int) -> float:
    return float(resupply_count)


def evaluate(
    chromosome: List[int],
    points: List[Point],
    distance_matrix: DistanceMatrix,
    config: Config,
) -> EvaluationResult:
    errors = validate_chromosome(chromosome, points, config)
    if errors:
        return EvaluationResult(
            chromosome=list(chromosome),
            decoded_route=[],
            fitness=float("inf"),
            total_distance=0.0,
            priority_penalty=0.0,
            supply_penalty=0.0,
            resupply_count=0,
            is_valid=False,
            errors=errors,
        )

    decoded_route = decode_route(chromosome, points, distance_matrix, config)
    if not decoded_route.is_valid:
        return EvaluationResult(
            chromosome=list(chromosome),
            decoded_route=decoded_route.route,
            fitness=float("inf"),
            total_distance=decoded_route.total_distance,
            priority_penalty=0.0,
            supply_penalty=0.0,
            resupply_count=decoded_route.resupply_count,
            is_valid=False,
            errors=decoded_route.errors,
        )

    points_by_idx = {point.idx: point for point in points}
    priority_penalty = calculate_priority_penalty(chromosome, points_by_idx)
    supply_penalty = calculate_supply_penalty(decoded_route.resupply_count)
    fitness = (
        decoded_route.total_distance
        + config.lambda_priority * priority_penalty
        + config.lambda_supply * supply_penalty
    )

    return EvaluationResult(
        chromosome=list(chromosome),
        decoded_route=decoded_route.route,
        fitness=fitness,
        total_distance=decoded_route.total_distance,
        priority_penalty=priority_penalty,
        supply_penalty=supply_penalty,
        resupply_count=decoded_route.resupply_count,
        is_valid=True,
        errors=[],
    )
