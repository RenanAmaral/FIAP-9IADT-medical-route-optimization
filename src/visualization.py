from typing import List, Optional

import matplotlib.pyplot as plt

from src.data_loader import ORIGIN_TYPE, SUPPLY_TYPE
from src.models import EvolutionResult, Point


# Paleta validada (colorblind-safe, ver dataviz/validate_palette.js, modo light)
COLOR_BEST = "#2a78d6"
COLOR_AVERAGE = "#eb6834"
COLOR_ORIGIN = "#0b0b0b"
COLOR_SUPPLY = "#2a78d6"
COLOR_ROUTE = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#52514e"

# Prioridade: status vermelho/amber/verde + tamanho (encoding secundario)
PRIORITY_COLORS = {"ALTA": "#d03b3b", "MEDIA": "#b87700", "BAIXA": "#0ca30c"}
PRIORITY_SIZES = {"ALTA": 260, "MEDIA": 170, "BAIXA": 90}
MARKER_EDGE = "#0b0b0b"


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
    """Mapa da rota final: origem, hospitais (cor/tamanho por prioridade) e abastecimentos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    points_by_idx = {point.idx: point for point in points}

    # Linha da rota (recessiva, atras dos marcadores)
    route_x = [points_by_idx[idx].lon for idx in decoded_route]
    route_y = [points_by_idx[idx].lat for idx in decoded_route]
    ax.plot(route_x, route_y, color=COLOR_ROUTE, linewidth=1.2, zorder=1, label="Rota")

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
    ax.set_title("Rota final (origem, hospitais por prioridade e abastecimentos)")
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
