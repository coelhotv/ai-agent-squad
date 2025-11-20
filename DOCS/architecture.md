# Architecture

## Hybrid Stack

The app is purposely split between a **native LLM host** and a **containerized Python backend**:

1. **LLM Host (macOS):** Ollama runs outside Docker to keep GPU access to the M2’s Metal cores. The default model is `deepseek-r1:14b-qwen-distill-q4_K_M`, and the UI communicates with it via `http://host.docker.internal:11434`.
2. **Agent Backend (Docker):** FastAPI + LangGraph + persistence modules live inside the `app` service defined in `docker-compose.yml`. This keeps dependencies clean and replicable.
3. **Bridge:** `app.py` talks to Ollama through the Docker special DNS name provided by Docker Desktop; no VPNs or extra proxies are needed.

## Persistence Layers

- **`tasks.db`:** Stores what you see on `/tasks_dashboard` and `tasks.html`. Each row represents a task (status, research, PRD, stories, pending approval). `Task` models are defined in `app.py` and kept in sync with the UI.
- **`checkpoints.sqlite`:** Maintained by `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`. It captures the LangGraph state machine’s detailed context so graph executions (research → PRD → stories → future nodes) can pause/resume reliably.
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

## Research & Agent Tooling

- **Perplexity:** When `PERPLEXITY_API_KEY` is set, the Research node calls Perplexity’s `sonar-pro` model to get structured JSON output (summary, opportunities, risks, references). Formatting helpers live in `app.py`.
- **DuckDuckGo Fallback:** Absent a key or if Perplexity fails, the Research node uses `ddgs` through `DDGS.text`/`DDGS.news` to build a fallback summary.
- **Product Agent:** Uses Ollama to write PRDs and user stories. Approved research becomes input for the next node.
- **Structured Ollama Calls:** `call_ollama_json` posts to the Ollama `/api/generate` endpoint with a JSON schema per agent (PRD, stories, flows, wireframes). The model is forced to return valid structured JSON, which we unwrap into text artifacts before persisting. UX artifacts share context: the wireframe generator receives the Mermaid flow plus user stories to keep both deliverables consistent.
- **Future Agents:** UX, Engineering, QA, and GTM nodes will continue the serialized path (each pausing for human approvals), keeping the same checkpoint + DB record pattern so everything remains auditable.

## Monitoring & Logging

- Python’s standard `logging` module reports graph progress, errors, and approvals with timestamps.
- SQLAlchemy engine logs are throttled (`logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)`) to keep dashboard polling quiet.
- Docker logs (`docker-compose logs app_service`) surface agent progress, especially those `logger.info` statements inside each LangGraph node.

## Static Assets

- `index.html` (task intake) and `tasks.html` (live dashboard) are served directly via FastAPI. The intake page renders the workflow timeline, live status pill, artifact collapsibles, and UX preview buttons that open new tabs for Mermaid/Tailwind outputs. UI assets live right in the project root, which keeps deployment light and portable.

See `DOCS/setup.md` for environment configuration and `DOCS/workflow.md` for how these pieces operate end-to-end.
