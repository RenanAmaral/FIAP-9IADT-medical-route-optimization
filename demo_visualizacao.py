from dataclasses import replace

from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.genetic_algorithm import run_genetic_algorithm
from src.visualization import plot_fitness_evolution, plot_route_map


def main() -> None:
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    config = replace(DEFAULT_CONFIG, seed=42)

    evolution = run_genetic_algorithm(points, distance_matrix, config)
    best = evolution.best

    print(f"Geracoes: {evolution.generations}")
    print(f"Melhor cromossomo: {best.chromosome}")
    print(f"Melhor rota: {best.decoded_route}")
    print(f"Distancia total: {best.total_distance:.2f} km")
    print(f"Reabastecimentos: {best.resupply_count}")
    print(f"Melhor fitness: {best.fitness:.2f}")

    plot_fitness_evolution(evolution, save_path="fitness.png")
    plot_route_map(points, best.decoded_route, save_path="route.png")
    print()
    print("Imagens salvas: fitness.png e route.png")


if __name__ == "__main__":
    main()
