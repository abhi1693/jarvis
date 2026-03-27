import asyncio

from app.main import interact
from app.models import InteractionRequest


def test_interact_is_gated_when_admin_is_not_visible(monkeypatch):
    agent_called = {"value": False}

    async def fake_handle_interaction(*_args, **_kwargs):
        agent_called["value"] = True
        return None

    monkeypatch.setattr("app.main.perception_service.admin_visible_recently", lambda **_kwargs: False)
    monkeypatch.setattr("app.main.agent.handle_interaction", fake_handle_interaction)

    response = asyncio.run(interact(InteractionRequest(message="hello", modality="text")))

    assert response["gated"] is True
    assert response["message"] == ""
    assert "Admin not visible" in response["detail"]
    assert agent_called["value"] is False
