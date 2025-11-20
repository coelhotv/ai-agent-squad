# Operations & Monitoring

## Monitoring the System

- Watch Docker logs with `docker-compose logs -f app_service` to view LangGraph progress statements (`logger.info` calls in `research_node`, `product_prd_node`, etc.).
- Use `docker stats app_service` if you suspect resource contention.
- Health-check the UI endpoints at `http://localhost:8000` and `http://localhost:8000/tasks_dashboard`.
- On startup the intake page queries `/get_pending_approval` and locks the submission form if a previous workflow is still waiting, so you must finish that work before starting a new idea.
- Checkpoint resilience: even if the backend restarts, `checkpoints.sqlite` keeps pending graph state so approvals can resume where they left off.

## Dashboard & CSV Export

- `/tasks_dashboard` (`tasks.html`) renders every row from `tasks.db`. It automatically polls `/tasks` every five seconds and shows statuses with color coding.
- The **Export CSV** button hits `GET /tasks/export` to stream `tasks_export.csv`, which includes columns for Task ID, Product Idea, Status, Research Summary, PRD Summary, User Stories, and Pending Approval Content. Use it for external audits or backups.
- For manual data inspection, open `tasks.db` using `sqlite3 tasks.db "SELECT * FROM Task;"`.

## Logs & Debugging

- FastAPI logs appear in Docker output, showing request lifecycle and approval events.
- SQLAlchemy logging is throttled to `WARNING` level (`logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)`) to minimize noise from frequent dashboard polling.
- At startup the app logs a configuration summary (database URL, checkpoint path, Ollama host/model, Perplexity key presence, `/data` writability, and Ollama connectivity). Check `docker-compose logs` for these preflight messages to confirm the environment matches expectations.
- Use `docker-compose down` to stop the app, then `docker-compose up --build` if you’ve made code changes that require a rebuild.
- If you edit `app.py`, restart the service (`docker-compose restart app_service`) or run `uvicorn app:app --reload` for rapid iteration outside Docker.

## Database Hygiene

- Remove `tasks.db` or `checkpoints.sqlite` when you need to reset the system or switch graph state definitions. Always stop the Docker service before deleting them to avoid SQLite locks.
- Backups can simply be zipped copies of both files. For longer-lived data, consider periodically copying them to a `backups/` folder with timestamps.

## Operations Notes

- Approvals are the single source of truth for the graph state. When a task pauses, the UI, `tasks.db`, and `checkpoints.sqlite` all point to the same status label (`pending_research_approval`, `pending_prd_approval`, etc.).
- Runtime configuration (Perplexity API key, Ollama URL/model, DB paths) now lives in `.env`. Update that file and run `docker-compose up -d` to apply changes without editing code; keep secrets out of source control by only committing `.env.example`.
- For manual testing, you can post to `/respond_to_approval` with a JSON body like `{"task_id":"<id>", "approved":true}` if you need to simulate human approvals via Postman or curl.
- Document any future maintenance steps (e.g., upgrading dependencies in `requirements.txt`) inside this doc to keep the operations runbook centralized.
