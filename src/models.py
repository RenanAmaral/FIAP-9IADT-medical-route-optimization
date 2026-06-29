from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Point:
    idx: int
    type: str
    name: str
    lat: float
    lon: float
    priority: Optional[str]
    demand: int


@dataclass
class DecodedRoute:
    route: List[int]
    total_distance: float
    resupply_count: int
    remaining_load: int
    is_valid: bool = True
    errors: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass
class EvaluationResult:
    chromosome: List[int]
    decoded_route: List[int]
    fitness: float
    total_distance: float
    priority_penalty: float
    supply_penalty: float
    resupply_count: int
    is_valid: bool
    errors: List[str]


@dataclass(frozen=True)
class Config:
    vehicle_capacity: int
    lambda_priority: float
    lambda_supply: float
