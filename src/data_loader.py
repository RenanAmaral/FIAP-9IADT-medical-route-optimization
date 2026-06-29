import csv
from pathlib import Path
from typing import Iterable, List

from src.models import Point


VALID_PRIORITIES = {"ALTA", "MEDIA", "BAIXA"}
ORIGIN_TYPE = "origin"
HOSPITAL_TYPE = "hospital"
SUPPLY_TYPE = "supply"


def load_points(csv_path: str) -> List[Point]:
    path = Path(csv_path)
    points: List[Point] = []

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            priority = row["priority"].strip().upper() or None
            points.append(
                Point(
                    idx=int(row["idx"]),
                    type=row["type"].strip().lower(),
                    name=row["name"].strip(),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    priority=priority,
                    demand=int(row["demand"]),
                )
            )

    validate_points(points)
    return points


def validate_points(points: Iterable[Point]) -> None:
    point_list = list(points)
    origins = [point for point in point_list if point.type == ORIGIN_TYPE]
    hospitals = [point for point in point_list if point.type == HOSPITAL_TYPE]
    supply_stations = [point for point in point_list if point.type == SUPPLY_TYPE]

    if len(origins) != 1:
        raise ValueError("Catalog must contain exactly one origin.")
    if not hospitals:
        raise ValueError("Catalog must contain at least one hospital.")
    if not supply_stations:
        raise ValueError("Catalog must contain at least one supply station.")

    seen_ids = set()
    for point in point_list:
        if point.idx in seen_ids:
            raise ValueError(f"Duplicated point idx: {point.idx}.")
        seen_ids.add(point.idx)

        if point.type not in {ORIGIN_TYPE, HOSPITAL_TYPE, SUPPLY_TYPE}:
            raise ValueError(f"Invalid point type for idx {point.idx}: {point.type}.")

        if point.type == HOSPITAL_TYPE:
            if point.demand <= 0:
                raise ValueError(f"Hospital {point.idx} must have demand greater than zero.")
            if point.priority not in VALID_PRIORITIES:
                raise ValueError(f"Hospital {point.idx} must have a valid priority.")

        if point.type == SUPPLY_TYPE and point.demand != 0:
            raise ValueError(f"Supply station {point.idx} must have zero demand.")

        if point.type == ORIGIN_TYPE and point.demand != 0:
            raise ValueError("Origin must have zero demand.")


def get_origin(points: List[Point]) -> Point:
    origins = [point for point in points if point.type == ORIGIN_TYPE]
    if len(origins) != 1:
        raise ValueError("Catalog must contain exactly one origin.")
    return origins[0]


def get_hospitals(points: List[Point]) -> List[Point]:
    return [point for point in points if point.type == HOSPITAL_TYPE]


def get_supply_stations(points: List[Point]) -> List[Point]:
    return [point for point in points if point.type == SUPPLY_TYPE]
