from dataclasses import replace

from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.fitness import evaluate
from src.llm_report import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    build_prompt,
    build_route_payload,
    generate_report,
    render_template_report,
)


def build_scenario():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    result = evaluate([3, 1, 5, 2, 4], points, distance_matrix, DEFAULT_CONFIG)
    return result, points, DEFAULT_CONFIG


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


class FakeClient:
    """Cliente Anthropic falso para testar o caminho LLM sem rede nem chave."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage("Relatorio gerado pela LLM.")


def test_payload_has_expected_fields():
    result, points, config = build_scenario()

    payload = build_route_payload(result, points, config)

    assert payload["total_distance_km"] == round(result.total_distance, 2)
    assert payload["resupply_count"] == result.resupply_count
    assert payload["hospital_order"] == result.chromosome
    assert payload["route"] == result.decoded_route
    assert payload["vehicle_capacity"] == config.vehicle_capacity


def test_payload_visit_position_only_counts_hospitals():
    result, points, config = build_scenario()

    payload = build_route_payload(result, points, config)

    hospital_stops = [stop for stop in payload["stops"] if stop["type"] == "hospital"]
    positions = [stop["visit_position"] for stop in hospital_stops]
    assert positions == list(range(1, len(hospital_stops) + 1))
    # estacoes de abastecimento e origem nao recebem visit_position
    assert all("visit_position" not in stop for stop in payload["stops"] if stop["type"] != "hospital")


def test_build_prompt_contains_data_and_json():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)

    prompt = build_prompt(payload)

    assert str(payload["total_distance_km"]) in prompt
    assert "JSON" in prompt
    assert "ALTA" in prompt


def test_system_prompt_has_anti_hallucination_rules():
    assert "APENAS os dados" in SYSTEM_PROMPT
    assert "NAO decide" in SYSTEM_PROMPT


def test_template_report_mentions_key_metrics():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)

    report = render_template_report(payload)

    assert f"{payload['total_distance_km']} km" in report
    assert "Reabastecimentos" in report
    assert "ALTA" in report


def test_generate_report_defaults_to_template_without_llm():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)

    report = generate_report(payload)

    assert report == render_template_report(payload)


def test_generate_report_uses_injected_client_when_llm_enabled():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)
    fake = FakeClient()

    report = generate_report(payload, use_llm=True, client=fake)

    assert report == "Relatorio gerado pela LLM."
    assert fake.calls[0]["model"] == DEFAULT_MODEL
    assert fake.calls[0]["system"] == SYSTEM_PROMPT
    assert str(payload["total_distance_km"]) in fake.calls[0]["messages"][0]["content"]
