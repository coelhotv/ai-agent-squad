# AI Product Squad Components

This document outlines the main components the AI Product Squad application, focusing on the agentic workflow managed by LangGraph.

## Core Components

- **FastAPI Backend (`app.py`):** Serves the API, manages task state in a SQLite database, and orchestrates the agent workflow.
- **LangGraph:** The core engine that defines and executes the sequence of agent tasks as a state machine.
- **Ollama:** Provides the LLM inference for the agents (e.g., `deepseek`, `qwen-coder`).
- **SQLite:** Used for two purposes:
  1. `tasks.db`: A simple application database to store the status and artifacts of each task for the UI.
  2. `checkpoints.sqlite`: The persistence layer for LangGraph, allowing workflows to be paused and resumed.
- **Vanilla JS Frontend (`index.html`, `tasks.html`):** A simple user interface for submitting ideas, providing human-in-the-loop (HITL) approvals, and viewing artifacts.

## Persistence Layers

- **`tasks.db`:** Stores what you see on `/tasks_dashboard` and `tasks.html`. Each row represents a task (status, research, PRD, stories, pending approval). `Task` models are defined in `app.py` and kept in sync with the UI.
- **`checkpoints.sqlite`:** Maintained by `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`. It captures the LangGraph state machine’s detailed context so graph executions (research → spec → code) can pause/resume reliably.
- **Environment resilience:** Engine uses `sqlite` with `check_same_thread=False`, and hooks `ensure_column` to evolve schema without migrations, so you can add optional fields (`research_summary`, `prd_summary`, `user_stories`) safely.
- **Configuration:** All tunables (DB URL, checkpoint path, Ollama host/model, Perplexity API info) are loaded via `app_settings.py` (`pydantic-settings`). Update `.env` to change values without touching code; startup logs verify what the app actually loaded.

## API Surface

Key FastAPI endpoints:

- `/start_task` (POST): ingests a product idea, creates a DB record, and starts the LangGraph workflow with a unique `task_id`.
- `/respond_to_approval` (POST): resumes paused graph once a human approves.
- `/update_artifacts` (POST): persists human edits to research/PRD/stories/flow/wireframe while the status stays pending, so the updated text is stored for previews and later approval.
- `/tasks` (GET): returns every task record for the dashboard.
- `/tasks/export` (GET): streams all records as a downloadable CSV (`tasks_export.csv`) for offline audits.
- `/tasks_dashboard` and `/index.html`: static UI entry points served via `FileResponse`.

## Agent Tooling

- **Research Agent:** When `PERPLEXITY_API_KEY` is set, the Research node calls Perplexity’s `sonar-pro` model to get structured JSON output (summary, opportunities, risks, references). Formatting helpers live in `app.py`.
  - **DuckDuckGo Fallback:** Absent a key or if Perplexity fails, the Research node uses `ddgs` through `DDGS.text`/`DDGS.news` to build a fallback summary.
- **Product Agent:** Uses Ollama to write PRDs and user stories. Approved product artifacts becomes input for the next node.
- **UX Agent:** Uses Ollama to 
- **Architect Agent:** Uses Ollama to 
- **Speac QA Agent:** Uses Ollama to 
- **Developer Agent:** Uses Ollama to 
- **QA Agent:** Uses Ollama to 
- **Structured Ollama Calls:** `call_ollama_json` posts to the Ollama `/api/generate` endpoint with a JSON schema per agent (PRD, stories, flows, wireframes). The model is forced to return valid structured JSON, which we unwrap into text artifacts before persisting. UX artifacts share context: the wireframe generator receives the Mermaid flow plus user stories to keep both deliverables consistent.
- **Future Agents:** GTM nodes will continue the serialized path (each pausing for human approvals), keeping the same checkpoint + DB record pattern so everything remains auditable.

## Monitoring & Logging

- Python’s standard `logging` module reports graph progress, errors, and approvals with timestamps.
- SQLAlchemy engine logs are throttled (`logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)`) to keep dashboard polling quiet.
- Docker logs (`docker-compose logs app_service`) surface agent progress, especially those `logger.info` statements inside each LangGraph node.

## Static Assets

- `index.html` (task intake) and `tasks.html` (live dashboard) are served directly via FastAPI. The intake page renders the workflow timeline, live status pill, artifact collapsibles, and UX preview buttons that open new tabs for Mermaid/Tailwind outputs. UI assets live right in the project root, which keeps deployment light and portable.

See `DOCS/setup.md` for environment configuration and `DOCS/workflow.md` for how these pieces operate end-to-end.
