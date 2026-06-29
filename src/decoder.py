from typing import List, Tuple

from src.data_loader import get_hospitals, get_origin, get_supply_stations
from src.distance import DistanceMatrix, get_distance
from src.models import Config, DecodedRoute, Point


def validate_chromosome(chromosome: List[int], points: List[Point], config: Config) -> List[str]:
    errors: List[str] = []
    points_by_idx = {point.idx: point for point in points}
    hospital_ids = {point.idx for point in get_hospitals(points)}

    if len(chromosome) != len(set(chromosome)):
        errors.append("Chromosome cannot contain duplicated hospitals.")

    for gene in chromosome:
        point = points_by_idx.get(gene)
        if point is None:
            errors.append(f"Chromosome contains unknown point idx: {gene}.")
            continue
        if point.type != "hospital":
            errors.append(f"Chromosome can contain only hospitals. Invalid idx: {gene}.")

    for hospital in get_hospitals(points):
        if hospital.demand > config.vehicle_capacity:
            errors.append(
                f"Hospital {hospital.idx} demand ({hospital.demand}) is greater than vehicle capacity "
                f"({config.vehicle_capacity})."
            )

    invalid_ids = [gene for gene in chromosome if gene not in hospital_ids]
    if invalid_ids:
        errors.append(f"Chromosome contains non-hospital ids: {invalid_ids}.")

    return _unique_errors(errors)


def decode_route(
    chromosome: List[int],
    points: List[Point],
    distance_matrix: DistanceMatrix,
    config: Config,
) -> DecodedRoute:
    errors = validate_chromosome(chromosome, points, config)
    origin = get_origin(points)

    if errors:
        return DecodedRoute(
            route=[origin.idx],
            total_distance=0.0,
            resupply_count=0,
            remaining_load=config.vehicle_capacity,
            is_valid=False,
            errors=errors,
        )

    points_by_idx = {point.idx: point for point in points}
    supply_station_ids = [point.idx for point in get_supply_stations(points)]

    route = [origin.idx]
    total_distance = 0.0
    current_idx = origin.idx
    current_load = config.vehicle_capacity
    resupply_count = 0

    for hospital_idx in chromosome:
        hospital = points_by_idx[hospital_idx]

        if current_load < hospital.demand:
            supply_idx = find_nearest_supply_station(current_idx, supply_station_ids, distance_matrix)
            total_distance += get_distance(current_idx, supply_idx, distance_matrix)
            route.append(supply_idx)
            current_idx = supply_idx
            current_load = config.vehicle_capacity
            resupply_count += 1

        total_distance += get_distance(current_idx, hospital.idx, distance_matrix)
        route.append(hospital.idx)
        current_idx = hospital.idx
        current_load -= hospital.demand

    total_distance += get_distance(current_idx, origin.idx, distance_matrix)
    route.append(origin.idx)

    return DecodedRoute(
        route=route,
        total_distance=total_distance,
        resupply_count=resupply_count,
        remaining_load=current_load,
        is_valid=True,
        errors=[],
    )


def find_nearest_supply_station(
    current_idx: int,
    supply_station_ids: List[int],
    distance_matrix: DistanceMatrix,
) -> int:
    if not supply_station_ids:
        raise ValueError("At least one supply station is required.")

    return min(
        supply_station_ids,
        key=lambda supply_idx: get_distance(current_idx, supply_idx, distance_matrix),
    )


def _unique_errors(errors: List[str]) -> List[str]:
    unique: List[str] = []
    for error in errors:
        if error not in unique:
            unique.append(error)
    return unique
