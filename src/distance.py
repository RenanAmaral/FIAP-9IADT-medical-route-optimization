from math import sqrt
from typing import Dict, List, Tuple

from src.models import Point


DistanceMatrix = Dict[Tuple[int, int], float]


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def build_distance_matrix(points: List[Point]) -> DistanceMatrix:
    distance_matrix: DistanceMatrix = {}
    for origin in points:
        for destination in points:
            distance_matrix[(origin.idx, destination.idx)] = euclidean_distance(
                origin.lat,
                origin.lon,
                destination.lat,
                destination.lon,
            )
    return distance_matrix


def get_distance(from_idx: int, to_idx: int, distance_matrix: DistanceMatrix) -> float:
    try:
        return distance_matrix[(from_idx, to_idx)]
    except KeyError as exc:
        raise KeyError(f"Distance not found for pair ({from_idx}, {to_idx}).") from exc
