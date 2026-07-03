import json
from pathlib import Path
from typing import Iterator, List, Optional

import matplotlib.pyplot as plt

from src.data_loader import ORIGIN_TYPE, SUPPLY_TYPE
from src.models import EvolutionResult, Point


# Paleta validada (colorblind-safe, ver dataviz/validate_palette.js, modo light)
COLOR_BEST = "#2a78d6"
COLOR_AVERAGE = "#eb6834"
COLOR_ORIGIN = "#0b0b0b"
COLOR_SUPPLY = "#2a78d6"
COLOR_ROUTE = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#52514e"

# Contorno do mapa do Brasil (estados)
BRAZIL_GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "brasil_estados.geojson"
COLOR_BR_FILL = "#eef1ec"
COLOR_BR_BORDER = "#b7c0b0"

# Prioridade: status vermelho/amber/verde + tamanho (encoding secundario)
PRIORITY_COLORS = {"ALTA": "#d03b3b", "MEDIA": "#b87700", "BAIXA": "#0ca30c"}
PRIORITY_SIZES = {"ALTA": 200, "MEDIA": 120, "BAIXA": 70}
MARKER_EDGE = "#0b0b0b"


def _iter_polygons(geometry: dict) -> Iterator[list]:
    """Itera os poligonos de uma geometria GeoJSON (Polygon ou MultiPolygon)."""
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        yield from geometry["coordinates"]


def draw_brazil(ax: plt.Axes, geojson_path: Path = BRAZIL_GEOJSON_PATH) -> None:
    """Desenha os estados do Brasil como fundo do mapa (nada se o arquivo faltar)."""
    path = Path(geojson_path)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for feature in data["features"]:
        for polygon in _iter_polygons(feature["geometry"]):
            exterior = polygon[0]  # anel externo do poligono
            xs = [point[0] for point in exterior]
            ys = [point[1] for point in exterior]
            ax.fill(xs, ys, facecolor=COLOR_BR_FILL, edgecolor=COLOR_BR_BORDER, linewidth=0.6, zorder=0)


def plot_fitness_evolution(
    evolution: EvolutionResult,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Grafico de convergencia: melhor fitness e fitness media por geracao."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    generations = [record.generation for record in evolution.history]
    best = [record.best_fitness for record in evolution.history]
    average = [record.average_fitness for record in evolution.history]

    ax.plot(generations, best, color=COLOR_BEST, linewidth=2, label="Melhor fitness")
    ax.plot(generations, average, color=COLOR_AVERAGE, linewidth=2, linestyle="--", label="Fitness media")

    ax.set_xlabel("Geracao", color=COLOR_TEXT)
    ax.set_ylabel("Fitness (menor e melhor)", color=COLOR_TEXT)
    ax.set_title("Evolucao da fitness por geracao")
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend()

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_route_map(
    points: List[Point],
    decoded_route: List[int],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Mapa da rota final sobre o Brasil: origem, hospitais (cor/tamanho por prioridade) e abastecimentos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 10))

    # Mapa do Brasil ao fundo
    draw_brazil(ax)

    points_by_idx = {point.idx: point for point in points}

    # Linha da rota (sobre o mapa, atras dos marcadores)
    route_x = [points_by_idx[idx].lon for idx in decoded_route]
    route_y = [points_by_idx[idx].lat for idx in decoded_route]
    ax.plot(route_x, route_y, color=COLOR_ROUTE, linewidth=1.0, alpha=0.8, zorder=1, label="Rota")

    seen_priorities: set = set()
    seen_supply = False
    for point in points:
        if point.type == ORIGIN_TYPE:
            ax.scatter(
                point.lon, point.lat, marker="*", s=420, color=COLOR_ORIGIN,
                edgecolors=MARKER_EDGE, linewidths=0.8, zorder=3, label="Origem",
            )
        elif point.type == SUPPLY_TYPE:
            ax.scatter(
                point.lon, point.lat, marker="s", s=150, color=COLOR_SUPPLY,
                edgecolors=MARKER_EDGE, linewidths=0.8, zorder=3,
                label=None if seen_supply else "Abastecimento",
            )
            seen_supply = True
        else:
            priority = point.priority
            label = None if priority in seen_priorities else f"Hospital {priority}"
            seen_priorities.add(priority)
            ax.scatter(
                point.lon, point.lat, marker="o", s=PRIORITY_SIZES[priority],
                color=PRIORITY_COLORS[priority], edgecolors=MARKER_EDGE, linewidths=0.8,
                zorder=2, label=label,
            )

    ax.set_xlabel("Longitude", color=COLOR_TEXT)
    ax.set_ylabel("Latitude", color=COLOR_TEXT)
    ax.set_title("Rota final no mapa do Brasil (origem, hospitais por prioridade e abastecimentos)")
    ax.set_xlim(-75, -32)
    ax.set_ylim(-34, 7)
    ax.set_aspect(1.03)  # reduz a distorcao lon/lat na latitude do Brasil
    ax.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
