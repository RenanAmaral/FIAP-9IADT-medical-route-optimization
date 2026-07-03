# FIAP-9IADT-medical-route-optimization

Otimizacao da rota de um veiculo de distribuicao de insumos medicos (TSP medico)
com **algoritmo genetico**, em **cidades reais do Brasil**. O veiculo parte de Brasilia,
visita 25 hospitais em capitais/grandes cidades respeitando a ordem de prioridade das
entregas (ALTA > MEDIA > BAIXA) e reabastece em estacoes quando a carga acaba. A rota
final e plotada sobre o **mapa do Brasil**.

## O que foi implementado

- catalogo de pontos (origem, hospitais e abastecimento) e loader com validacao;
- calculo de distancia **Haversine** (em km);
- **decoder** de cromossomo para rota real, com abastecimento automatico;
- funcao de **fitness** (distancia + prioridade + reabastecimento);
- **algoritmo genetico** completo: populacao inicial, selecao por torneio, Order
  Crossover (OX), mutacao swap/inversion, elitismo e loop evolutivo;
- **baselines** (rota aleatoria e nearest neighbor) para comparacao justa;
- **experimentos** E1/E2/E3 com tabela comparativa e metricas;
- **visualizacao**: curva de convergencia e mapa da rota;
- **integracao com LLM** (Gemini por padrao): relatorio, instrucoes ao motorista,
  relatorio de eficiencia, sugestoes de melhoria e perguntas em linguagem natural (Q&A);
- **67 testes** automatizados;
- notebook demonstrativo ponta a ponta.

Relatorio tecnico completo em [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md).

## Resultados (seed=42, 25 hospitais em cidades reais)

| Metodo | Distancia (km) | Reab | PosCrit | Fitness |
|--------|----------------|------|---------|---------|
| Rota aleatoria | 49 826 | 6 | 16,12 | 53 391 |
| Nearest neighbor | 22 028 | 6 | 16,00 | 25 473 |
| **Algoritmo genetico** (E2) | 17 015 | 6 | **11,62** | **20 310** |

`PosCrit` = posicao media de visita dos 8 hospitais ALTA (menor = atendidos mais cedo).
O GA ganha ~62% em fitness vs aleatoria e ~20% vs nearest neighbor, e atende os
hospitais urgentes mais cedo (posicao media ~11 vs ~16 dos baselines) mesmo em escala
nacional, onde varias cidades ALTA sao distantes.

## Estrutura

```text
data/
  pontos_entrega.csv       # 25 hospitais + origem + 6 abastecimentos (cidades reais)
  brasil_estados.geojson   # contorno dos estados para o mapa
src/
  config.py            # parametros centralizados
  data_loader.py       # catalogo + validacao
  distance.py          # Haversine + matriz de distancias
  decoder.py           # cromossomo -> rota real (insere abastecimento)
  fitness.py           # avaliacao (distancia + prioridade + reabastecimento)
  baseline.py          # rota aleatoria e nearest neighbor
  genetic_algorithm.py # operadores + loop evolutivo
  experiments.py       # E1/E2/E3 + tabela comparativa
  visualization.py     # convergencia e mapa da rota
  llm_report.py        # payload + relatorio (template / LLM)
  models.py            # dataclasses
tests/
  test_*.py            # 61 testes
notebooks/
  otimizacao_rotas_medicas.ipynb   # pipeline ponta a ponta
docs/
  relatorio_tecnico.md
demo_avaliacao_rotas.py            # demo da funcao evaluate()
demo_visualizacao.py               # roda o GA e salva os graficos
requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python -m pytest -q                 # roda os 61 testes
python demo_avaliacao_rotas.py      # avalia um cromossomo manual
python demo_visualizacao.py         # roda o GA e salva fitness.png / route.png
```

Notebook ponta a ponta:

```bash
jupyter notebook notebooks/otimizacao_rotas_medicas.ipynb
```

## Uso rapido

Rodar o algoritmo genetico:

```python
from dataclasses import replace
from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.genetic_algorithm import run_genetic_algorithm

points = load_points("data/pontos_entrega.csv")
distance_matrix = build_distance_matrix(points)
config = replace(DEFAULT_CONFIG, seed=42)  # seed fixa = reproduzivel

evolution = run_genetic_algorithm(points, distance_matrix, config)
best = evolution.best
print(best.decoded_route)   # rota final (com origem e abastecimentos)
print(best.total_distance)  # em km
print(best.fitness)
```

Avaliar um cromossomo pronto:

```python
from src.fitness import evaluate

# apenas indices de hospitais; origem e abastecimento entram automaticamente
result = evaluate([3, 1, 5, 2, 4], points, distance_matrix, config)
print(result.fitness, result.decoded_route)
```

## Relatorio via LLM (opcional)

O relatorio deterministico funciona offline. Para gerar via LLM, o provider padrao e o
**Google Gemini** (free tier). Crie a chave em https://aistudio.google.com e defina:

```bash
pip install google-genai       # ja incluso no requirements
# Windows PowerShell:
$env:GEMINI_API_KEY = "sua-chave"
```

A mesma integracao cobre os quatro usos do item 3 do desafio:

```python
from src.llm_report import (
    build_route_payload, generate_report, generate_driver_instructions,
    answer_question, suggest_improvements,
    build_efficiency_payload, generate_efficiency_report,
)

payload = build_route_payload(best, points, config)

print(generate_report(payload))                       # template deterministico (offline)
print(generate_report(payload, use_llm=True))         # relatorio via Gemini
print(generate_driver_instructions(payload))          # instrucoes para o motorista
print(answer_question(payload, "Quantos reabastecimentos?"))  # Q&A em linguagem natural
print(suggest_improvements(payload))                  # sugestoes de melhoria

# relatorio de eficiencia usa os experimentos:
from src.experiments import run_all_experiments
resultados = run_all_experiments(points, distance_matrix, config)
print(generate_efficiency_report(build_efficiency_payload(resultados)))

# provider alternativo:
print(generate_report(payload, use_llm=True, provider="anthropic"))  # Claude (ANTHROPIC_API_KEY)
```

## Regras consideradas

- o veiculo parte do Hospital Central e sempre retorna a origem;
- todos os hospitais sao visitados; cada um tem demanda e prioridade;
- a carga inicial vem de `vehicle_capacity`;
- quando a carga nao cobre a proxima entrega, o decoder insere o abastecimento mais
  proximo e recarrega ao maximo;
- a fitness considera distancia total, penalidade de prioridade e de reabastecimento;
- o cromossomo contem apenas hospitais — origem e abastecimento sao inseridos pelo
  decoder, o que mantem os operadores geneticos classicos sempre validos.
