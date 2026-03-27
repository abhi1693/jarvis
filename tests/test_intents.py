import asyncio

from app.intents import IntentService


class FakeLLMAdapter:
    enabled = False


def test_rule_intent_extracts_person_and_objective_facts():
    service = IntentService(FakeLLMAdapter())
    result = asyncio.run(service.parse("My name is Asha and I'm working on a local multimodal agent."))

    assert result.name == "remember"
    assert result.extracted_facts["name"] == "Asha"
    assert "current_focus" in result.extracted_facts
    assert any(candidate.category == "person" for candidate in result.memory_candidates)


def test_rule_intent_extracts_generic_operating_context():
    service = IntentService(FakeLLMAdapter())
    result = asyncio.run(
        service.parse(
            "Your job is to adapt to my priorities, and you should avoid hardcoded duties."
        )
    )

    assert result.name == "orient"
    assert any(
        candidate.category == "charter" and candidate.title == "duty"
        for candidate in result.memory_candidates
    )
    assert any(candidate.category == "constraint" for candidate in result.memory_candidates)
