from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import AgentRuntime
from app.config import get_settings
from app.intents import IntentService
from app.llm import LLMAdapter
from app.media import MediaService
from app.memory_store import MemoryStore
from app.models import (
    InteractionRequest,
    ObservationRequest,
    ProfileRequest,
    SystemStateResponse,
    ToolInvocationRequest,
)
from app.perception import PerceptionService
from app.self_improvement import SelfImprovementService
from app.tools.filesystem import FilesystemTool
from app.tools.shell import ShellTool
from app.tools.web_search import WebSearchTool


settings = get_settings()
memory_store = MemoryStore(settings.db_path)
llm_adapter = LLMAdapter(settings)
intent_service = IntentService(llm_adapter)
filesystem_tool = FilesystemTool(settings.repo_root)
shell_tool = ShellTool(settings.repo_root, settings.command_timeout_seconds)
web_search_tool = WebSearchTool()
self_improvement_service = SelfImprovementService(
    settings.repo_root,
    memory_store,
    filesystem_tool,
    shell_tool,
)
perception_service = PerceptionService(settings.snapshot_dir, settings.admin_face_path)
media_service = MediaService(settings.media_dir)
agent = AgentRuntime(
    settings=settings,
    memory_store=memory_store,
    intent_service=intent_service,
    llm_adapter=llm_adapter,
    fs_tool=filesystem_tool,
    shell_tool=shell_tool,
    web_search_tool=web_search_tool,
    self_improvement=self_improvement_service,
)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(settings.static_dir) / "index.html")


@app.get("/api/state", response_model=SystemStateResponse)
async def get_state() -> SystemStateResponse:
    return SystemStateResponse(
        app_name=settings.app_name,
        llm_enabled=settings.llm_enabled,
        repo_root=str(settings.repo_root),
        memory_counts=memory_store.get_memory_counts(),
        runtime_context=memory_store.list_context_memories(limit=6),
        recent_memories=memory_store.recall(limit=6),
        recent_skills=memory_store.list_recent_skills(limit=6),
        recent_insights=memory_store.list_recent_insights(limit=6),
        recent_interactions=memory_store.list_recent_interactions(limit=8),
        last_observation=memory_store.get_last_observation(),
    )


@app.get("/api/memories")
async def list_memories(category: str | None = None, query: str = "") -> dict[str, object]:
    return {"items": memory_store.recall(query=query, category=category, limit=20)}


@app.get("/api/skills")
async def list_skills() -> dict[str, object]:
    return {"items": memory_store.list_recent_skills(limit=20)}


@app.get("/api/insights")
async def list_insights() -> dict[str, object]:
    return {"items": memory_store.list_recent_insights(limit=20)}


@app.get("/api/interactions")
async def list_interactions() -> dict[str, object]:
    return {"items": memory_store.list_recent_interactions(limit=20)}


@app.post("/api/interactions")
async def interact(payload: InteractionRequest) -> dict[str, object]:
    text = (payload.message or payload.note or "").strip()
    media_path = None
    if payload.audio_data_url:
        try:
            media_path = media_service.save_audio_data_url(payload.audio_data_url)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    if not text and not media_path:
        raise HTTPException(status_code=400, detail="Interaction requires text, note, or audio.")

    if not text and media_path:
        text = "audio note captured without transcript"

    response = await agent.handle_interaction(
        text,
        modality=payload.modality,
        metadata=payload.metadata,
        media_path=media_path,
    )
    return response.model_dump()


@app.post("/api/chat")
async def chat_alias(payload: InteractionRequest) -> dict[str, object]:
    payload.modality = payload.modality or "text"
    return await interact(payload)


@app.post("/api/observe")
async def observe(payload: ObservationRequest) -> dict[str, object]:
    try:
        observation = perception_service.analyze_snapshot(payload.image_data_url, payload.note)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    observation_id = memory_store.store_observation(
        admin_present=observation["admin_present"],
        face_count=observation["face_count"],
        brightness=observation["brightness"],
        note=observation["note"],
        image_path=observation["image_path"],
    )
    summary = (
        payload.note.strip()
        if payload.note
        else (
            f"camera observation: present={observation['admin_present']}, "
            f"admin={observation['admin_detected']}, faces={observation['face_count']}, "
            f"brightness={observation['brightness']}"
        )
    )
    memory_store.record_interaction(
        "sensor",
        summary,
        "observe",
        modality="camera",
        channel="sensor",
        metadata=observation,
        media_path=observation["image_path"],
    )
    observation["id"] = observation_id
    return observation


@app.post("/api/vision/enroll")
async def enroll_admin_face(payload: ObservationRequest) -> dict[str, object]:
    if not payload.image_data_url:
        raise HTTPException(status_code=400, detail="image_data_url is required")

    try:
        return perception_service.enroll_admin(payload.image_data_url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/profile")
async def update_profile(payload: ProfileRequest) -> dict[str, str]:
    agent.update_profile(payload.model_dump())
    return {"status": "ok"}


@app.post("/api/evolution/scan")
async def run_evolution_scan() -> dict[str, object]:
    return self_improvement_service.scan()


@app.post("/api/self-improvement/scan")
async def run_self_scan_alias() -> dict[str, object]:
    return await run_evolution_scan()


@app.post("/api/tools/invoke")
async def invoke_tool(payload: ToolInvocationRequest) -> dict[str, object]:
    return agent.invoke_tool(payload.tool, payload.args)
