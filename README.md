# FIAP-9IADT-medical-route-optimization

Medical route optimization using Genetic Algorithm for hospital supply distribution.

## Pessoa 1 - Dominio e fitness

Esta parte do projeto implementa a base do problema para o TSP medico:

- catalogo de pontos de entrega, origem e abastecimento;
- calculo de distancia por Haversine;
- decoder de cromossomo para rota real com abastecimento automatico;
- funcao de fitness para o algoritmo genetico consumir.

O algoritmo genetico completo nao esta implementado aqui. A Pessoa 2 pode usar a funcao `evaluate()` para avaliar individuos representados como `list[int]`.

## Estrutura

```text
data/
  pontos_entrega.csv
src/
  config.py
  data_loader.py
  decoder.py
  distance.py
  fitness.py
  models.py
tests/
  test_decoder.py
  test_distance.py
  test_fitness.py
notebooks/
  demo_pessoa_1.ipynb
demo_pessoa_1.py
requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python demo_pessoa_1.py
python -m pytest -q
```

Para abrir a demonstracao em notebook:

```bash
jupyter notebook notebooks/demo_pessoa_1.ipynb
```

## Interface para a Pessoa 2

```python
from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.fitness import evaluate

points = load_points("data/pontos_entrega.csv")
distance_matrix = build_distance_matrix(points)

chromosome = [3, 1, 5, 2, 4]
result = evaluate(chromosome, points, distance_matrix, DEFAULT_CONFIG)

print(result.fitness)
print(result.decoded_route)
```

O cromossomo deve conter apenas os indices das unidades hospitalares obrigatorias. A origem e as unidades de abastecimento sao inseridas automaticamente pelo decoder.
