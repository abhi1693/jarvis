import asyncio

from app.intents import IntentService


class FakeLLMAdapter:
    enabled = False


def test_rule_intent_extracts_profile_facts():
    service = IntentService(FakeLLMAdapter())
    result = asyncio.run(service.parse("My name is Asha and I'm working on a local multimodal agent."))

    assert result.name == "remember"
    assert result.extracted_facts["name"] == "Asha"
    assert "current_focus" in result.extracted_facts
