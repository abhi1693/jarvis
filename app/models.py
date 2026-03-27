from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    category: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.8


class IntentResult(BaseModel):
    name: str
    confidence: float = 0.0
    extracted_facts: dict[str, str] = Field(default_factory=dict)
    suggested_tools: list[str] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)


class InteractionRequest(BaseModel):
    message: str = Field(default="", max_length=10_000)
    modality: str = Field(default="text")
    note: str | None = None
    audio_data_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationRequest(BaseModel):
    image_data_url: str | None = None
    note: str | None = None


class ProfileRequest(BaseModel):
    name: str | None = None
    role: str | None = None
    goals: str | None = None
    preferences: str | None = None


class ToolInvocationRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    id: int
    category: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source: str
    confidence: float
    valid_from: str | None = None
    valid_until: str | None = None
    created_at: str


class SkillRecord(BaseModel):
    id: int
    name: str
    description: str
    trigger_hint: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    success_count: int
    created_at: str
    last_used_at: str | None = None


class EvolutionInsightRecord(BaseModel):
    id: int
    severity: str
    source: str
    title: str
    details: str
    file_path: str | None = None
    line_number: int | None = None
    status: str
    created_at: str


class ObservationRecord(BaseModel):
    id: int
    admin_present: bool
    face_count: int
    brightness: float
    note: str | None = None
    image_path: str | None = None
    created_at: str


class InteractionRecord(BaseModel):
    id: int
    role: str
    content: str
    intent: str
    modality: str
    channel: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    media_path: str | None = None
    created_at: str


class InteractionResponse(BaseModel):
    message: str
    intent: IntentResult
    memories: list[MemoryRecord] = Field(default_factory=list)
    skills: list[SkillRecord] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    insights: list[EvolutionInsightRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemStateResponse(BaseModel):
    app_name: str
    llm_enabled: bool
    repo_root: str
    memory_counts: dict[str, int]
    runtime_context: list[MemoryRecord]
    recent_memories: list[MemoryRecord]
    recent_skills: list[SkillRecord]
    recent_insights: list[EvolutionInsightRecord]
    recent_interactions: list[InteractionRecord]
    last_observation: ObservationRecord | None = None
