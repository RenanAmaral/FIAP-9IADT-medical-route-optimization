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
- **Capacidade e penalizada sobretudo pela distancia**: o desvio ate a estacao mais
  proxima ja entra na `distancia_total` — no dataset nacional, cada reabastecimento custa
  **400 a 600 km reais**. Esse e o mecanismo dominante.
- **Custo explicito por parada** (desvio deliberado do plano original): alem da distancia,
  a fitness soma `λ_abastecimento × numero_de_reabastecimentos`. A justificativa e que uma
  parada de reabastecimento tem um custo proprio — tempo de carga, mao de obra, janela de
  operacao — que nao se reduz aos quilometros percorridos. Ver a analise de sensibilidade
  na secao 5.
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

O catalogo tem **25 hospitais + 1 origem + 6 estacoes de abastecimento (32 pontos)** em
**cidades reais do Brasil** (capitais e grandes cidades), com a origem em Brasilia
([data/pontos_entrega.csv](../data/pontos_entrega.csv)). As distancias Haversine ficam em
km reais (centenas a milhares de km), e a demanda total (~570) contra a capacidade de 100
forca **6 reabastecimentos**, tornando o problema realista.

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
| E1 | 50 / 100 / 5% | Aleatoria | 49 826 | 6 | 16,12 | 53 391 |
| E1 | | Nearest neighbor | 22 028 | 6 | 16,00 | 25 473 |
| E1 | | **Algoritmo genetico** | 19 450 | 6 | 16,62 | 22 855 |
| E2 | 100 / 200 / 10% | **Algoritmo genetico** | 17 015 | 6 | **11,62** | **20 310** |
| E3 | 200 / 300 / 15% | **Algoritmo genetico** | 17 359 | 6 | 11,00 | 20 604 |

`PosCrit` = posicao media de visita dos hospitais ALTA (menor = atendidos mais cedo);
ha 8 hospitais ALTA entre os 25. Tempo do E3 ~3,5s (vs 0,3s do E1).

### Leitura dos resultados

- **Ganho do GA**: ~62% em fitness vs rota aleatoria e ~20% vs nearest neighbor (E2).
- **Prioridade funciona**: os baselines atendem os hospitais ALTA na posicao media ~16
  (tarde, pois varias cidades ALTA sao distantes — Manaus, Boa Vista, Porto Velho); o GA
  (E2/E3) puxa essa media para **~11**, atendendo as urgencias mais cedo mesmo pagando
  distancia. E a prova de que a `penalidade_prioridade` cumpre seu papel na escala nacional.
- **Efeito das geracoes**: o E1 (pequeno) prioriza distancia e quase nao melhora a
  `PosCrit` (16,62); com mais geracoes (E2/E3) o GA equilibra distancia e prioridade.
- **Retorno decrescente / ruido**: E3 (mut 15%) nao supera o E2 (17 359 vs 17 015 km) —
  a mutacao alta explora mais mas atrapalha a convergencia no numero fixo de geracoes.
  **E2 e o melhor custo-beneficio.**

A melhor rota (E2) parte de Brasilia, cobre as 25 cidades com 6 reabastecimentos e
percorre **17 015 km**. Ela e plotada sobre o **mapa do Brasil** por `plot_route_map`.

### Sensibilidade: qual o peso real do termo de abastecimento?

O plano original previa que a capacidade fosse penalizada *apenas* pela distancia extra.
Optamos por manter tambem um custo explicito por parada (`λ_abastecimento = 10`). Medindo
o efeito dessa decisao:

- **O termo discrimina, mas pouco.** O numero de reabastecimentos varia entre solucoes:
  em 2000 rotas aleatorias, 274 usaram 5 paradas e 1726 usaram 6. Entao o termo de fato
  empurra para menos paradas — nao e uma constante inerte.
- **A distancia domina por ~50x.** Cada parada custa **10** na fitness (0,05% de ~20 310),
  enquanto o desvio fisico ate a estacao custa **400–600 km** reais (2–3% da fitness). O
  mecanismo dominante continua sendo o natural, como o plano previa.
- **Zerar o termo nao melhora a rota.** Rodando com `λ_abastecimento = 0`, o GA encontrou
  uma rota *pior* sob os dois criterios (com e sem o termo) — a diferenca ficou dentro do
  ruido entre seeds (~2%), nao foi efeito do termo.

**Conclusao:** o custo por parada funciona como um *desempate* de baixo impacto, alinhado
com a realidade operacional (uma parada consome tempo e mao de obra), sem distorcer a
otimizacao — que continua guiada pela distancia e pela prioridade.

## 6. Integracao com LLM

O modulo [src/llm_report.py](../src/llm_report.py) serializa o resultado em um payload
limpo e usa uma LLM para quatro finalidades (item 3 do desafio):

1. **Relatorio da rota** — `generate_report` (tambem tem um template deterministico
   offline, usado como padrao/fallback sem chave);
2. **Instrucoes para o motorista** e a equipe de entrega — `generate_driver_instructions`;
3. **Relatorio de eficiencia** comparando os metodos ao longo dos experimentos
   (economia de distancia/tempo/recursos e ganho percentual) — `generate_efficiency_report`;
4. **Sugestoes de melhoria** no processo — `suggest_improvements`;
5. **Perguntas em linguagem natural** sobre a rota (Q&A) — `answer_question`.

Todas compartilham um nucleo (`_call_llm`) e o mesmo **system prompt com controle de
alucinacao**: a LLM usa apenas os dados fornecidos e nao decide nem altera a rota.

- **Provider padrao**: **Google Gemini** (free tier), com alternativa Anthropic Claude
  (`provider="anthropic"`).
- **Prompts eficientes**: cada tarefa monta um prompt especifico (`build_*_prompt`) sobre
  o payload JSON, pedindo exatamente o formato de saida desejado.

## 7. Como reproduzir

```bash
pip install -r requirements.txt
python -m pytest -q                 # 61 testes
python demo_visualizacao.py         # roda o GA e salva fitness.png / route.png
jupyter notebook notebooks/otimizacao_rotas_medicas.ipynb   # pipeline ponta a ponta
```

## 8. Checklist do desafio

- [x] Dataset com cidades reais do Brasil (25 hospitais + origem + 6 abastecimentos) — **obrigatorio**
- [x] Calculo de distancia entre pontos (Haversine, km reais) — **obrigatorio**
- [x] Algoritmo genetico para TSP — **obrigatorio**
- [x] Fitness com distancia e prioridade — **obrigatorio**
- [x] Selecao, crossover e mutacao — **obrigatorio**
- [x] Comparacao com baseline (aleatoria e nearest neighbor) — **obrigatorio**
- [x] Grafico de convergencia e mapa da rota sobre o mapa do Brasil — **obrigatorio**
- [x] LLM gerando relatorio, instrucoes ao motorista, relatorio de eficiencia,
  sugestoes de melhoria e Q&A em linguagem natural — **obrigatorio**
- [x] Capacidade / reabastecimento — *opcional* (contemplado pelo decoder)
