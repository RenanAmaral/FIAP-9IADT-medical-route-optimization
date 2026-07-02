from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.experiments import (
    DEFAULT_EXPERIMENTS,
    ExperimentSpec,
    average_critical_position,
    format_comparison_table,
    run_all_experiments,
    run_experiment,
)


SMALL_SPECS = [
    ExperimentSpec("T1", pop_size=20, generations=10, mutation_rate=0.05),
    ExperimentSpec("T2", pop_size=30, generations=12, mutation_rate=0.10),
]


def load_scenario():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    return points, distance_matrix


def test_default_experiments_match_pdf_specs():
    assert [(spec.name, spec.pop_size, spec.generations, spec.mutation_rate) for spec in DEFAULT_EXPERIMENTS] == [
        ("E1", 50, 100, 0.05),
        ("E2", 100, 200, 0.10),
        ("E3", 200, 300, 0.15),
    ]


def test_average_critical_position_counts_only_alta_hospitals():
    points, _ = load_scenario()

    # hospitais ALTA: 1, 4, 7, 10. No cromossomo [3, 1, 5, 4] estao nas posicoes 2 e 4.
    position = average_critical_position([3, 1, 5, 4], points)

    assert position == (2 + 4) / 2


def test_average_critical_position_is_zero_without_alta():
    points, _ = load_scenario()

    # 3 (BAIXA), 2 (MEDIA), 6 (BAIXA): nenhum ALTA
    assert average_critical_position([3, 2, 6], points) == 0.0


def test_run_experiment_collects_all_metrics():
    points, distance_matrix = load_scenario()
    spec = SMALL_SPECS[0]

    result = run_experiment(spec, points, distance_matrix, DEFAULT_CONFIG)

    for metrics in (result.random, result.nearest_neighbor, result.genetic):
        assert metrics.distance > 0
        assert metrics.fitness > 0
        assert metrics.runtime_seconds >= 0
        assert metrics.avg_critical_position > 0
    assert len(result.evolution.history) == spec.generations


def test_ga_does_not_lose_to_nearest_neighbor_in_experiment():
    points, distance_matrix = load_scenario()

    result = run_experiment(SMALL_SPECS[0], points, distance_matrix, DEFAULT_CONFIG)

    assert result.genetic.fitness <= result.nearest_neighbor.fitness
    assert result.gain_vs_nearest_pct >= 0


def test_run_all_experiments_runs_each_spec():
    points, distance_matrix = load_scenario()

    results = run_all_experiments(points, distance_matrix, DEFAULT_CONFIG, specs=SMALL_SPECS)

    assert [result.spec.name for result in results] == ["T1", "T2"]


def test_format_comparison_table_lists_methods_and_experiments():
    points, distance_matrix = load_scenario()
    results = run_all_experiments(points, distance_matrix, DEFAULT_CONFIG, specs=SMALL_SPECS)

    table = format_comparison_table(results)

    assert "Aleatoria" in table
    assert "Nearest neighbor" in table
    assert "Algoritmo genetico" in table
    assert "T1" in table and "T2" in table
    assert "Ganho GA" in table
