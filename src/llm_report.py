import json
from typing import Dict, List, Optional

from src.models import Config, EvaluationResult, Point


DEFAULT_MODEL = "claude-opus-4-8"

# Controle de alucinacao: a LLM apenas EXPLICA a rota, nao decide nem inventa dados.
SYSTEM_PROMPT = (
    "Voce e um assistente que explica, em portugues do Brasil e em linguagem natural, "
    "o resultado de uma rota de distribuicao medica ja calculada por um algoritmo genetico.\n"
    "Regras rigorosas:\n"
    "- Use APENAS os dados do JSON fornecido. Nao invente numeros, nomes ou pontos.\n"
    "- Voce NAO decide nem altera a rota; ela ja foi calculada. Apenas explique o resultado.\n"
    "- Se um dado nao estiver no JSON, diga que nao esta disponivel.\n"
    "- Cite distancia em km, reabastecimentos e prioridades exatamente como no JSON.\n"
    "- Responda somente com o relatorio final, sem expor raciocinio."
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


def generate_report(
    payload: Dict,
    use_llm: bool = False,
    client: Optional[object] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Gera o relatorio: template deterministico por padrao, LLM opcional (item 38)."""
    if not use_llm:
        return render_template_report(payload)

    if client is None:
        import anthropic  # import tardio: so exigido quando a LLM e usada

        client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(payload)}],
    )
    return "".join(block.text for block in message.content if block.type == "text")
