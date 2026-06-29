from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.fitness import evaluate


def main() -> None:
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    chromosome = [3, 1, 5, 2, 4]

    result = evaluate(chromosome, points, distance_matrix, DEFAULT_CONFIG)

    print(f"Cromossomo original: {result.chromosome}")
    print(f"Rota decodificada: {result.decoded_route}")
    print(f"Distancia total: {result.total_distance:.2f} km")
    print(f"Penalidade de prioridade: {result.priority_penalty:.2f}")
    print(f"Penalidade de abastecimento: {result.supply_penalty:.2f}")
    print(f"Fitness final: {result.fitness:.2f}")
    print(f"Numero de reabastecimentos: {result.resupply_count}")
    print(f"Valida: {result.is_valid}")
    if result.errors:
        print(f"Erros: {result.errors}")


if __name__ == "__main__":
    main()
