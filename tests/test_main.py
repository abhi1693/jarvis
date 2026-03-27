import asyncio

from fastapi import HTTPException

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


def test_interact_rejects_audio_without_transcript(monkeypatch):
    agent_called = {"value": False}

    async def fake_handle_interaction(*_args, **_kwargs):
        agent_called["value"] = True
        return None

    monkeypatch.setattr("app.main.perception_service.admin_visible_recently", lambda **_kwargs: True)
    monkeypatch.setattr("app.main.media_service.save_audio_data_url", lambda _data_url: "/tmp/audio.wav")
    monkeypatch.setattr("app.main.agent.handle_interaction", fake_handle_interaction)

    try:
        asyncio.run(
            interact(
                InteractionRequest(
                    message="",
                    note="",
                    modality="audio",
                    audio_data_url="data:audio/wav;base64,AAAA",
                )
            )
        )
        assert False, "Expected HTTPException for audio without transcript"
    except HTTPException as error:
        assert error.status_code == 400
        assert "without a transcript" in error.detail

    assert agent_called["value"] is False


def test_interact_accepts_audio_with_metadata_transcript(monkeypatch):
    captured = {}

    async def fake_handle_interaction(message, **kwargs):
        captured["message"] = message
        captured["kwargs"] = kwargs

        class DummyResponse:
            def model_dump(self):
                return {"message": "ok"}

        return DummyResponse()

    monkeypatch.setattr("app.main.perception_service.admin_visible_recently", lambda **_kwargs: True)
    monkeypatch.setattr("app.main.media_service.save_audio_data_url", lambda _data_url: "/tmp/audio.wav")
    monkeypatch.setattr("app.main.agent.handle_interaction", fake_handle_interaction)

    response = asyncio.run(
        interact(
            InteractionRequest(
                message="",
                note="",
                modality="audio",
                audio_data_url="data:audio/wav;base64,AAAA",
                metadata={"transcript": "schedule tomorrow at 10"},
            )
        )
    )

    assert response["message"] == "ok"
    assert captured["message"] == "schedule tomorrow at 10"
    assert captured["kwargs"]["media_path"] == "/tmp/audio.wav"
