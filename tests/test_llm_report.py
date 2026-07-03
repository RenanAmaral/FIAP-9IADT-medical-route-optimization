from dataclasses import replace

from src.config import DEFAULT_CONFIG
from src.data_loader import load_points
from src.distance import build_distance_matrix
from src.fitness import evaluate
from types import SimpleNamespace

from src.experiments import ExperimentSpec, run_all_experiments
from src.llm_report import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    SYSTEM_PROMPT,
    answer_question,
    build_efficiency_payload,
    build_prompt,
    build_route_payload,
    generate_driver_instructions,
    generate_efficiency_report,
    generate_report,
    render_template_report,
    suggest_improvements,
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


class FakeAnthropicClient:
    """Cliente Anthropic falso para testar o caminho LLM sem rede nem chave."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage("Relatorio gerado pela LLM.")


class FakeGeminiClient:
    """Cliente Gemini falso (espelha client.models.generate_content)."""

    def __init__(self):
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="Relatorio gerado pelo Gemini.")


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


def test_gemini_is_the_default_llm_provider():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)
    fake = FakeGeminiClient()

    report = generate_report(payload, use_llm=True, client=fake)

    assert report == "Relatorio gerado pelo Gemini."
    assert fake.calls[0]["model"] == DEFAULT_GEMINI_MODEL
    assert fake.calls[0]["config"]["system_instruction"] == SYSTEM_PROMPT
    assert str(payload["total_distance_km"]) in fake.calls[0]["contents"]


def test_anthropic_provider_uses_messages_api():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)
    fake = FakeAnthropicClient()

    report = generate_report(payload, use_llm=True, provider="anthropic", client=fake)

    assert report == "Relatorio gerado pela LLM."
    assert fake.calls[0]["model"] == DEFAULT_ANTHROPIC_MODEL
    assert fake.calls[0]["system"] == SYSTEM_PROMPT
    assert str(payload["total_distance_km"]) in fake.calls[0]["messages"][0]["content"]


def test_unknown_provider_raises():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)

    try:
        generate_report(payload, use_llm=True, provider="openai")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_driver_instructions_prompt_mentions_driver_and_route():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)
    fake = FakeGeminiClient()

    text = generate_driver_instructions(payload, client=fake)

    assert text == "Relatorio gerado pelo Gemini."
    contents = fake.calls[0]["contents"]
    assert "motorista" in contents.lower()
    assert str(payload["total_distance_km"]) in contents


def test_answer_question_includes_the_question():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)
    fake = FakeGeminiClient()

    answer_question(payload, "Quantos reabastecimentos foram feitos?", client=fake)

    assert "Quantos reabastecimentos foram feitos?" in fake.calls[0]["contents"]


def test_suggest_improvements_calls_llm():
    result, points, config = build_scenario()
    payload = build_route_payload(result, points, config)
    fake = FakeGeminiClient()

    text = suggest_improvements(payload, client=fake)

    assert text == "Relatorio gerado pelo Gemini."
    assert "melhoria" in fake.calls[0]["contents"].lower()


def test_efficiency_payload_and_report():
    points = load_points("data/pontos_entrega.csv")
    distance_matrix = build_distance_matrix(points)
    specs = [ExperimentSpec("T1", pop_size=20, generations=8, mutation_rate=0.1)]
    results = run_all_experiments(points, distance_matrix, DEFAULT_CONFIG, specs=specs)

    efficiency = build_efficiency_payload(results)
    assert efficiency["experimentos"][0]["experimento"] == "T1"
    assert "ganho_vs_nearest_pct" in efficiency["experimentos"][0]

    fake = FakeGeminiClient()
    text = generate_efficiency_report(efficiency, client=fake)
    assert text == "Relatorio gerado pelo Gemini."
    assert "eficiencia" in fake.calls[0]["contents"].lower()
