import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.genetic_algorithm import run_genetic_algorithm
from src.models import Config
from src.visualization import plot_fitness_evolution, plot_route_map


def small_config(**overrides):
    base = dict(
        vehicle_capacity=100, lambda_priority=5.0, lambda_supply=10.0,
        pop_size=20, generations=8, seed=42,
    )
    base.update(overrides)
    return Config(**base)


def run_small_ga():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    result = run_genetic_algorithm(points, distance_matrix, small_config())
    return points, result


def test_plot_fitness_evolution_draws_two_series():
    _, evolution = run_small_ga()

    ax = plot_fitness_evolution(evolution)

    assert len(ax.lines) == 2
    labels = [line.get_label() for line in ax.lines]
    assert "Melhor fitness" in labels
    assert "Fitness media" in labels
    plt.close(ax.figure)


def test_plot_fitness_evolution_saves_file(tmp_path):
    _, evolution = run_small_ga()
    output = tmp_path / "fitness.png"

    plot_fitness_evolution(evolution, save_path=str(output))

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_route_map_returns_axes_with_route_line():
    points, evolution = run_small_ga()

    ax = plot_route_map(points, evolution.best.decoded_route)

    assert ax.lines  # a linha da rota foi desenhada
    assert ax.collections  # marcadores (scatter) foram desenhados
    plt.close(ax.figure)


def test_plot_route_map_saves_file(tmp_path):
    points, evolution = run_small_ga()
    output = tmp_path / "route.png"

    plot_route_map(points, evolution.best.decoded_route, save_path=str(output))

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_route_map_legend_differentiates_point_types():
    points, evolution = run_small_ga()

    ax = plot_route_map(points, evolution.best.decoded_route)

    legend_labels = {text.get_text() for text in ax.get_legend().get_texts()}
    assert "Origem" in legend_labels
    assert "Abastecimento" in legend_labels
    assert any(label.startswith("Hospital") for label in legend_labels)
    plt.close(ax.figure)
