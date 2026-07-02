# Relatorio Tecnico — TSP Medico com Algoritmo Genetico

Tech Challenge — Fase 2. Otimizacao da rota de um veiculo de distribuicao de insumos
medicos que parte de um hospital central, visita todas as unidades hospitalares
respeitando a ordem de prioridade das entregas e reabastece quando a carga acaba.

## 1. O problema

Trata-se de uma variante do **Caixeiro Viajante (TSP)** com tres restricoes adicionais:

1. passar por **todas** as unidades hospitalares (pontos obrigatorios);
2. considerar a **ordem de prioridade** das entregas (ALTA > MEDIA > BAIXA);
3. respeitar a **capacidade de carga** do veiculo, reabastecendo em estacoes de
   abastecimento sempre que a carga nao cobre a proxima entrega.

O algoritmo genetico calcula a rota. A LLM entra apenas depois, para explicar o
resultado em linguagem natural — ela nao decide a rota.

### Variaveis do problema

| Variavel | Papel no modelo |
|----------|-----------------|
| Distancia | Custo base entre dois pontos (Haversine, em km) |
| Prioridade | Peso da entrega: ALTA=3, MEDIA=2, BAIXA=1 |
| Demanda | Carga que cada unidade hospitalar consome |
| Capacidade de carga | Limite do veiculo; ao nao cobrir a proxima entrega, reabastece |
| Estacoes de abastecimento | Pontos onde a carga volta ao maximo |

## 2. Modelagem

A decisao central do projeto e **isolar toda a complexidade das restricoes no decoder
do fitness**, mantendo o cromossomo simples. Assim os operadores geneticos classicos
(OX, swap/inversion) continuam validos sem nunca gerar rota invalida.

- **Cromossomo** = permutacao **apenas** das unidades hospitalares. Origem e estacoes
  de abastecimento **nao** entram no cromossomo.
- **Reabastecimento inserido automaticamente** durante a avaliacao ([src/decoder.py](../src/decoder.py)):
  ao percorrer o cromossomo, se a carga atual e menor que a demanda do proximo hospital,
  o decoder insere a estacao de abastecimento mais proxima e recarrega ao maximo.
- **Capacidade nao vira penalidade artificial**: o desvio ate a estacao ja entra na
  distancia total. Logo a capacidade e penalizada *naturalmente* pela distancia extra.
- **Prioridade vira penalidade soft**: `penalidade_prioridade = Σ (peso_prioridade × posicao_de_visita)`.
  Minimizar essa soma empurra as entregas ALTA para o comeco da rota.
- **Fitness** = `distancia_total + λ_prioridade × penalidade_prioridade + λ_abastecimento × penalidade_abastecimento`
  ([src/fitness.py](../src/fitness.py)). Com `DEFAULT_CONFIG`: `λ_prioridade = 5.0`, `λ_abastecimento = 10.0`.

### Estruturas de dados

| Camada | Estrutura | Contem origem? | Contem abastecimento? | Quem mexe |
|--------|-----------|----------------|-----------------------|-----------|
| Catalogo | lista fixa de `Point` | sim (idx 0) | sim | ninguem (constante) |
| Cromossomo | `list[int]` (permutacao) | nao | nao | crossover / mutacao |
| Populacao | `list[list[int]]` | — | — | selecao / elitismo |
| Rota (fenotipo) | `list[int]` | sim | sim (inserido) | decoder no fitness |

O catalogo tem 12 hospitais + 1 origem + 3 estacoes de abastecimento (16 pontos),
gerado em [data/pontos_entrega.csv](../data/pontos_entrega.csv) e calibrado para
forcar ~2 a 3 reabastecimentos.

## 3. Algoritmo genetico

Implementado em [src/genetic_algorithm.py](../src/genetic_algorithm.py):

- **Populacao inicial**: `pop_size` permutacoes aleatorias; opcionalmente semeia 1
  individuo com a rota do *nearest neighbor* (isso garante que o GA nunca sai pior
  que essa heuristica).
- **Selecao por torneio**: sorteia `tournament_size` competidores e devolve o de
  menor fitness.
- **Order Crossover (OX)**: preserva um segmento do pai 1 e completa com o pai 2 na
  ordem, garantindo permutacao valida.
- **Mutacao swap/inversion**: aplicada com probabilidade `mutation_rate`.
- **Elitismo**: os `n_elite` melhores passam intactos para a proxima geracao — isso
  torna o melhor fitness **monotonicamente nao-crescente**.
- **Loop evolutivo**: `avaliar → registrar → selecionar → cruzar → mutar → elitismo`
  por `generations` geracoes, com seed fixa para reprodutibilidade.

Os parametros ficam centralizados no `Config` ([src/config.py](../src/config.py)).

## 4. Baselines

Para comparacao justa ([src/baseline.py](../src/baseline.py)), duas referencias passam
pelo **mesmo decoder/fitness** do GA:

- **Rota aleatoria** — referencia fraca (permutacao embaralhada).
- **Nearest neighbor** — heuristica gulosa que parte da origem e sempre visita o
  hospital nao-visitado mais proximo.

## 5. Resultados

Experimentos com seed fixa (42), comparando aleatoria × nearest neighbor × GA
([src/experiments.py](../src/experiments.py)):

| Exp | Config (pop/ger/mut) | Metodo | Dist (km) | Reab | PosCrit | Fitness |
|-----|----------------------|--------|-----------|------|---------|---------|
| E1 | 50 / 100 / 5% | Aleatoria | 94,94 | 3 | 7,25 | 969,94 |
| E1 | | Nearest neighbor | 52,04 | 2 | 8,50 | 887,04 |
| E1 | | **Algoritmo genetico** | 60,03 | 2 | **2,50** | **700,03** |
| E2 | 100 / 200 / 10% | **Algoritmo genetico** | 56,55 | 2 | **2,50** | **696,55** |
| E3 | 200 / 300 / 15% | **Algoritmo genetico** | 56,16 | 2 | **2,50** | **696,16** |

`PosCrit` = posicao media de visita dos hospitais ALTA (menor = atendidos mais cedo).

### Leitura dos resultados

- **Ganho do GA**: ~28% em fitness vs rota aleatoria e ~21% vs nearest neighbor.
- **Prioridade funciona**: os baselines atendem os hospitais ALTA nas posicoes medias
  7,25 e 8,50 (tarde); o GA os coloca em **2,50** — os 4 hospitais urgentes sao
  atendidos essencialmente primeiro. E a prova numerica de que a `penalidade_prioridade`
  cumpre seu papel.
- **Trade-off correto**: o GA aceita distancia ligeiramente maior que o nearest
  neighbor (56,5 vs 52,0 km) em troca de fitness bem menor — trocou distancia por
  atender urgencias cedo, exatamente o comportamento desejado.
- **Retorno decrescente**: E2 e E3 convergem para praticamente a mesma solucao
  (696,55 vs 696,16), mas E3 leva ~4x mais tempo. E2 e o melhor custo-beneficio.

A melhor rota (E2):
`[0, 1, 4, 7, 101, 10, 8, 5, 101, 2, 11, 6, 3, 9, 12, 0]` — 56,55 km, 2 reabastecimentos.

## 6. Relatorio via LLM

O modulo [src/llm_report.py](../src/llm_report.py) serializa o resultado em um payload
limpo e gera um relatorio em linguagem natural. Ha dois caminhos:

- **template deterministico** (padrao, offline, sem chave);
- **LLM (Claude)** opcional, com controle de alucinacao no system prompt (usa apenas
  os dados do payload; nao decide nem altera a rota).

## 7. Como reproduzir

```bash
pip install -r requirements.txt
python -m pytest -q                 # 61 testes
python demo_visualizacao.py         # roda o GA e salva fitness.png / route.png
jupyter notebook notebooks/otimizacao_rotas_medicas.ipynb   # pipeline ponta a ponta
```

## 8. Checklist do desafio

- [x] Dataset simulado com 10–15 pontos — **obrigatorio**
- [x] Calculo de distancia entre pontos (Haversine) — **obrigatorio**
- [x] Algoritmo genetico para TSP — **obrigatorio**
- [x] Fitness com distancia e prioridade — **obrigatorio**
- [x] Selecao, crossover e mutacao — **obrigatorio**
- [x] Comparacao com baseline (aleatoria e nearest neighbor) — **obrigatorio**
- [x] Grafico de convergencia e mapa da rota — **obrigatorio**
- [x] LLM gerando relatorio em linguagem natural — **obrigatorio**
- [x] Capacidade / reabastecimento — *opcional* (contemplado pelo decoder)
