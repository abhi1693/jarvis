# Jarvis

Jarvis is a local-first multimodal agent scaffold. It learns from interaction across text, voice,
and camera, keeps durable memory, and evolves its own capabilities over time.

Its role is not supposed to be hardcoded. The current runtime is structured around operator-defined
context first: the user can shape duties, constraints, priorities, and collaboration style through
normal interaction, and the system stores that as durable operating context.

## What It Does

1. Multimodal interaction
   You can interact by typing, speaking, or being seen on camera.
2. Persistent memory
   The system stores operating context, user model details, experiences, learned skills,
   observations, and evolution insights in SQLite.
3. Skill accumulation
   When a tool-backed flow succeeds, Jarvis stores it as a reusable learned skill.
4. Evolution loop
   It can inspect its own repository, run checks, and record new insights about how to improve its
   environment.
5. Operator-shaped charter
   Duties are inferred from user-provided context rather than assumed at startup.
6. Optional LLM intent layer
   If an OpenAI-compatible model is configured, it improves intent parsing and response quality.
   Without it, the app still works with rule-based behavior.

## Architecture

### Backend

- `app/main.py`
  FastAPI API and route wiring.
- `app/agent.py`
  Multimodal interaction runtime, operating-context learning, and generic action routing.
- `app/memory_store.py`
  SQLite-backed persistence for operating context, memories, interactions, skills, observations,
  and insights.
- `app/intents.py`
  Rule-based and optional LLM-backed extraction of duties, constraints, preferences, and tasks.
- `app/perception.py`
  Camera frame analysis using OpenCV face detection.
- `app/media.py`
  Audio file persistence from browser-captured voice input.
- `app/self_improvement.py`
  Repository evolution scanning and insight capture.
- `app/tools/`
  Filesystem, shell, and web search tools.

### Frontend

- `app/static/index.html`
  Interface for vision, voice, text, operating context, memory, skills, and insights.
- `app/static/app.js`
  Browser camera capture, face overlays, admin enrollment, live mic interaction, and API calls.
- `app/static/styles.css`
  Minimal UI styling.

## Safety Model

- Filesystem access is rooted to the repository.
- Shell access uses an allowlist and blocks destructive commands.
- Evolution scans are bounded and record insights instead of performing unconstrained self-edits.
- Voice audio is stored locally.
- LLM use is optional and disabled unless configured.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Optional LLM

```bash
export LLM_COMPAT_URL="http://localhost:11434/v1"
export LLM_MODEL="your-model-name"
export LLM_API_KEY="optional"
```

## Current Limits

This is a solid multimodal MVP, not a finished autonomous organism.

- Voice understanding depends on browser speech recognition support unless you add a local STT
  backend later.
- Camera input currently detects presence, not identity.
- Camera now supports a simple local admin-face enrollment flow and draws dashed red/yellow/green boxes, but the recognition model is intentionally lightweight and not production-grade biometrics.
- Evolution is bounded to scanning, remembering, and safe local maintenance operations.
- There is no full autonomous planner yet for long-horizon self-directed tasks.
- Context extraction is generic but still heuristic in rule-based mode; a better local model will
  improve how well the agent infers duties and constraints from freeform instructions.

## Best Next Steps

1. Add a local speech-to-text backend so voice works without browser support.
2. Add identity recognition and temporal behavior summaries on top of camera observations.
3. Add change-set generation plus explicit approval so the system can edit itself safely.
4. Add memory consolidation and pattern extraction for habits and routines across many sessions.
