import json
from typing import Dict, List, Optional

from src.models import Config, EvaluationResult, Point


# Gemini 2.5 Flash: rapido e disponivel no free tier do Google AI Studio.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# Haiku 4.5: opcao paga (Anthropic), a mais barata dessa familia.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"

# Controle de alucinacao: a LLM apenas explica/instrui/responde com base nos dados,
# nunca decide nem inventa. Vale para todas as tarefas (relatorio, instrucoes, Q&A...).
SYSTEM_PROMPT = (
    "Voce e um assistente que responde, em portugues do Brasil e em linguagem natural, "
    "sobre rotas de distribuicao medica ja calculadas por um algoritmo genetico.\n"
    "Regras rigorosas:\n"
    "- Use APENAS os dados fornecidos (JSON). Nao invente numeros, nomes ou pontos.\n"
    "- Voce NAO decide nem altera a rota; ela ja foi calculada. Apenas use os dados.\n"
    "- Se um dado nao estiver no JSON, diga que nao esta disponivel.\n"
    "- Cite distancia em km, reabastecimentos e prioridades exatamente como no JSON.\n"
    "- Seja direto e util; responda apenas o que foi pedido, sem expor raciocinio."
)


def build_route_payload(
    result: EvaluationResult,
    points: List[Point],
    config: Config,
) -> Dict:
    """Serializa o resultado da rota em um payload limpo (item 36)."""
    points_by_idx = {point.idx: point for point in points}
    stops: List[Dict] = []
    visit_position = 0

    for idx in result.decoded_route:
        point = points_by_idx[idx]
        stop: Dict = {"idx": point.idx, "name": point.name, "type": point.type}
        if point.type == "hospital":
            visit_position += 1
            stop["priority"] = point.priority
            stop["demand"] = point.demand
            stop["visit_position"] = visit_position
        stops.append(stop)

    return {
        "vehicle_capacity": config.vehicle_capacity,
        "total_distance_km": round(result.total_distance, 2),
        "resupply_count": result.resupply_count,
        "priority_penalty": round(result.priority_penalty, 2),
        "supply_penalty": round(result.supply_penalty, 2),
        "fitness": round(result.fitness, 2),
        "is_valid": result.is_valid,
        "hospital_order": list(result.chromosome),
        "route": list(result.decoded_route),
        "stops": stops,
    }


def build_prompt(payload: Dict) -> str:
    """Monta o prompt do usuario a partir do payload (item 37)."""
    return (
        "Explique o resultado desta rota de distribuicao medica para um gestor. "
        "Baseie-se SOMENTE nestes dados (JSON):\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Escreva um relatorio curto cobrindo: (1) resumo da rota e distancia total em km; "
        "(2) como os hospitais de prioridade ALTA foram atendidos; "
        "(3) reabastecimentos realizados; (4) conclusao sobre a qualidade da solucao."
    )


def render_template_report(payload: Dict) -> str:
    """Relatorio deterministico, sem LLM (item 38 - caminho padrao/offline)."""
    lines = ["RELATORIO DA ROTA DE DISTRIBUICAO MEDICA", ""]
    lines.append(f"Capacidade do veiculo: {payload['vehicle_capacity']}")
    lines.append(f"Distancia total: {payload['total_distance_km']} km")
    lines.append(f"Reabastecimentos: {payload['resupply_count']}")
    lines.append(f"Fitness final: {payload['fitness']}")
    lines.append(f"Rota {'valida' if payload['is_valid'] else 'invalida'}.")
    lines.append("")

    high_priority = [stop for stop in payload["stops"] if stop.get("priority") == "ALTA"]
    if high_priority:
        atendimentos = ", ".join(
            f"{stop['name']} (posicao {stop['visit_position']})" for stop in high_priority
        )
        lines.append(f"Hospitais de prioridade ALTA atendidos: {atendimentos}.")
    else:
        lines.append("Nenhum hospital de prioridade ALTA nesta rota.")

    supply_stops = [stop for stop in payload["stops"] if stop["type"] == "supply"]
    if supply_stops:
        nomes = ", ".join(stop["name"] for stop in supply_stops)
        lines.append(f"Paradas de reabastecimento: {nomes}.")

    lines.append("")
    lines.append("Sequencia de visita:")
    lines.append(" -> ".join(stop["name"] for stop in payload["stops"]))
    return "\n".join(lines)


def _gemini_text(system_prompt: str, user_prompt: str, client: Optional[object], model: str) -> str:
    if client is None:
        from google import genai  # import tardio: so exigido quando a LLM e usada

        client = genai.Client()  # le GEMINI_API_KEY ou GOOGLE_API_KEY do ambiente

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={"system_instruction": system_prompt},
    )
    return response.text


def _anthropic_text(
    system_prompt: str, user_prompt: str, client: Optional[object], model: str, thinking: Optional[Dict]
) -> str:
    if client is None:
        from anthropic import Anthropic  # import tardio

        client = Anthropic()  # le ANTHROPIC_API_KEY do ambiente

    request = {
        "model": model,
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if thinking is not None:
        request["thinking"] = thinking

    message = client.messages.create(**request)
    return "".join(block.text for block in message.content if block.type == "text")


def _call_llm(
    user_prompt: str,
    provider: str = "gemini",
    client: Optional[object] = None,
    model: Optional[str] = None,
    thinking: Optional[Dict] = None,
) -> str:
    """Nucleo compartilhado: envia SYSTEM_PROMPT + user_prompt ao provider escolhido."""
    if provider == "gemini":
        return _gemini_text(SYSTEM_PROMPT, user_prompt, client, model or DEFAULT_GEMINI_MODEL)
    if provider == "anthropic":
        return _anthropic_text(SYSTEM_PROMPT, user_prompt, client, model or DEFAULT_ANTHROPIC_MODEL, thinking)
    raise ValueError(f"Provider LLM desconhecido: {provider!r}. Use 'gemini' ou 'anthropic'.")


def generate_report(
    payload: Dict,
    use_llm: bool = False,
    provider: str = "gemini",
    client: Optional[object] = None,
    model: Optional[str] = None,
    thinking: Optional[Dict] = None,
) -> str:
    """Relatorio da rota: template deterministico por padrao, LLM opcional (item 38)."""
    if not use_llm:
        return render_template_report(payload)
    return _call_llm(build_prompt(payload), provider, client, model, thinking)


# --- Instrucoes para motoristas e equipes de entrega ------------------------------

def build_driver_instructions_prompt(payload: Dict) -> str:
    return (
        "Gere instrucoes passo a passo para o motorista e a equipe de entrega seguirem "
        "esta rota. Use SOMENTE estes dados (JSON):\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Para cada parada, na ordem da rota, diga o que fazer (entregar no hospital ou "
        "reabastecer), o nome do ponto e, nas entregas, a prioridade e a demanda. "
        "Destaque claramente os reabastecimentos e finalize com o retorno a origem."
    )


def generate_driver_instructions(
    payload: Dict,
    provider: str = "gemini",
    client: Optional[object] = None,
    model: Optional[str] = None,
    thinking: Optional[Dict] = None,
) -> str:
    """Instrucoes detalhadas para o motorista/equipe seguirem a rota."""
    return _call_llm(build_driver_instructions_prompt(payload), provider, client, model, thinking)


# --- Sugestoes de melhoria no processo --------------------------------------------

def build_improvement_prompt(payload: Dict) -> str:
    return (
        "Com base nos dados desta rota, sugira melhorias no processo de distribuicao "
        "(por exemplo: reduzir reabastecimentos, atender prioridades mais cedo, revisar "
        "a capacidade do veiculo). Baseie-se SOMENTE nestes dados (JSON):\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Liste de 3 a 5 sugestoes objetivas, cada uma justificada pelos dados."
    )


def suggest_improvements(
    payload: Dict,
    provider: str = "gemini",
    client: Optional[object] = None,
    model: Optional[str] = None,
    thinking: Optional[Dict] = None,
) -> str:
    """Sugestoes de melhoria no processo, com base nos padroes da rota."""
    return _call_llm(build_improvement_prompt(payload), provider, client, model, thinking)


# --- Perguntas em linguagem natural (Q&A) -----------------------------------------

def build_question_prompt(payload: Dict, question: str) -> str:
    return (
        "Responda a pergunta usando SOMENTE estes dados (JSON):\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Pergunta: {question}\n"
        "Se a resposta nao estiver nos dados, diga que nao esta disponivel."
    )


def answer_question(
    payload: Dict,
    question: str,
    provider: str = "gemini",
    client: Optional[object] = None,
    model: Optional[str] = None,
    thinking: Optional[Dict] = None,
) -> str:
    """Responde perguntas em linguagem natural sobre a rota."""
    return _call_llm(build_question_prompt(payload, question), provider, client, model, thinking)


# --- Relatorio de eficiencia (usa os experimentos E1/E2/E3) -----------------------

def build_efficiency_payload(experiment_results: List) -> Dict:
    """Resume os experimentos (aleatoria x nearest neighbor x GA) em um payload limpo."""
    rows = []
    for result in experiment_results:
        rows.append({
            "experimento": result.spec.name,
            "config": {
                "pop_size": result.spec.pop_size,
                "generations": result.spec.generations,
                "mutation_rate": result.spec.mutation_rate,
            },
            "aleatoria": {
                "distancia_km": round(result.random.distance, 2),
                "fitness": round(result.random.fitness, 2),
            },
            "nearest_neighbor": {
                "distancia_km": round(result.nearest_neighbor.distance, 2),
                "fitness": round(result.nearest_neighbor.fitness, 2),
            },
            "genetico": {
                "distancia_km": round(result.genetic.distance, 2),
                "fitness": round(result.genetic.fitness, 2),
                "reabastecimentos": result.genetic.resupply_count,
                "pos_media_prioridade_alta": round(result.genetic.avg_critical_position, 2),
                "tempo_s": round(result.genetic.runtime_seconds, 3),
            },
            "ganho_vs_aleatoria_pct": round(result.gain_vs_random_pct, 1),
            "ganho_vs_nearest_pct": round(result.gain_vs_nearest_pct, 1),
        })
    return {"experimentos": rows}


def build_efficiency_prompt(efficiency: Dict) -> str:
    return (
        "Escreva um relatorio de eficiencia das rotas comparando os tres metodos "
        "(aleatoria, nearest neighbor e algoritmo genetico) ao longo dos experimentos. "
        "Baseie-se SOMENTE nestes dados (JSON):\n\n"
        f"{json.dumps(efficiency, ensure_ascii=False, indent=2)}\n\n"
        "Aborde a economia de distancia (proxy de tempo e recursos), o ganho percentual "
        "do algoritmo genetico sobre as baselines e o atendimento das prioridades. "
        "Conclua qual configuracao tem o melhor custo-beneficio."
    )


def generate_efficiency_report(
    efficiency: Dict,
    provider: str = "gemini",
    client: Optional[object] = None,
    model: Optional[str] = None,
    thinking: Optional[Dict] = None,
) -> str:
    """Relatorio de eficiencia das rotas (economia de tempo e recursos)."""
    return _call_llm(build_efficiency_prompt(efficiency), provider, client, model, thinking)
