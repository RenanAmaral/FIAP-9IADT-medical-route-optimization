from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Tuple

from src.models import Point


DistanceMatrix = Dict[Tuple[int, int], float]

EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia em km entre dois pontos (lat/lon em graus) sobre a superficie da Terra."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, (lat1, lon1, lat2, lon2))
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    inner = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(inner))


def build_distance_matrix(points: List[Point]) -> DistanceMatrix:
    distance_matrix: DistanceMatrix = {}
    for origin in points:
        for destination in points:
            distance_matrix[(origin.idx, destination.idx)] = haversine_distance(
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
