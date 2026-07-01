# FIAP-9IADT-medical-route-optimization

Base de avaliacao de rotas medicas para distribuicao de insumos hospitalares.

## O que foi implementado

O projeto ja possui a base do problema para o TSP medico:

- catalogo de pontos de entrega, origem e abastecimento;
- calculo de distancia euclidiana;
- decoder de cromossomo para rota real com abastecimento automatico;
- funcao de fitness para avaliar rotas;
- testes automatizados para distancia, decoder e fitness;
- demonstracao em script Python e notebook Jupyter.

A estrutura atual entrega a funcao `evaluate()` para avaliar cromossomos prontos representados como `list[int]`.

## O que falta fazer

Para completar o algoritmo genetico, ainda falta implementar:

- geracao automatica de cromossomos;
- criacao da populacao inicial;
- selecao dos melhores individuos;
- crossover entre rotas;
- mutacao de cromossomos;
- loop de evolucao por geracoes;
- escolha e exibicao da melhor rota encontrada.

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
  avaliacao_rotas_medicas.ipynb
demo_avaliacao_rotas.py
requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python demo_avaliacao_rotas.py
python -m pytest -q
```

Para abrir a demonstracao em notebook:

```bash
jupyter notebook notebooks/avaliacao_rotas_medicas.ipynb
```

## Como avaliar uma rota

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

## Regras consideradas

- o veiculo parte do Hospital Central;
- todos os hospitais do cromossomo sao visitados na ordem informada;
- cada hospital possui demanda e prioridade;
- a carga inicial vem da configuracao `vehicle_capacity`;
- quando a carga nao cobre a proxima entrega, o decoder insere o abastecimento mais proximo;
- ao abastecer, a carga volta para a capacidade maxima;
- a rota sempre retorna para a origem;
- a fitness considera distancia total, penalidade de prioridade e penalidade de abastecimento;
- a penalidade de abastecimento acompanha a quantidade de reabastecimentos realizados.
