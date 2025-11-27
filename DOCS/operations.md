# Operations & Monitoring

## Monitoring the Stack

- Tail the Docker service: `docker-compose logs -f app_service` shows the environment summary, per-node progress, QA verdicts, and Ollama/Perplexity connectivity checks (`log_environment_status`).
- Watch resource usage with `docker stats app_service` if agents stall.
- After startup, the intake UI calls `/get_pending_approval` so the system locks new submissions until the oldest pending approval clears. If a task is already paused, approve/reject it before sending a new idea.

## Workflow Interaction

- Approvals are the single source of truth. `/respond_to_approval` flips the `pending_*` status to `processing`, resumes the graph, and writes new artifacts back to `tasks.db`. Approve (`approved: true`) to continue, or reject (`approved: false`) to surface the resubmit banner. When a task starts in **semi-auto**, expect Research → PRD → Stories → UX to advance on their own; the UI’s live monitor (powered by `/tasks/{task_id}`) keeps the status card/artifacts refreshed until Spec Reasoning pauses for manual review.
- Humans can edit research/PRD/stories/flow/wireframe/spec/code/QA artifacts while the task is pending by clicking the Edit button; the overlay posts `/update_artifact`, updating both the DB and LangGraph checkpoint state.
- If you reject a review, use the resubmit banner in the UI or call `POST /resubmit_step` to rerun a single node. The backend clears downstream artifacts before rerunning the node function so no stale data leaks forward.
- `GET /get_pending_approval` is polled every 4 seconds while the UI waits for work and now skips tasks still auto-advancing. `/tasks` drives `tasks.html`, `/tasks/{task_id}` feeds the semi-auto monitor, and `/tasks/export` streams `tasks_export.csv` (includes all artifact columns plus `pending_approval_content`, `last_rejected_step`, `last_rejected_at`).

## Persistence Hygiene

- Stop the container (`docker-compose down`) before deleting `tasks.db` or `checkpoints.sqlite` to avoid SQLite locks. Removing them resets the entire queue. `docker-compose down -v` also removes the `app_data` volume that stores these files.
- Backup strategy: copy both SQLite files to a timestamped folder before destructive changes. The CSV export is lightweight, but the dual SQLite files capture all in-flight state.
- The app warns if the checkpoint data directory is missing or unwritable; it attempts to create the parent folder during startup (`log_environment_status`), so verify Docker can write to the mounted volume (`/data` in the container).

## Configuration & Env Vars

- Set `PERPLEXITY_API_KEY` to unlock sonar-pro research. Without it, the research agent falls back to DuckDuckGo (`ddgs`) and logs the fallback.
- Preference order: `OLLAMA_REASONING_MODEL` for planning/QA, `OLLAMA_CODING_MODEL` for UX/engineering/code, `OLLAMA_MODEL` as a legacy fallback. Use `OLLAMA_BASE_URL` if Ollama runs on a nonstandard host; the app defaults to `http://host.docker.internal:11434`.
- Override `DATABASE_URL` and `CHECKPOINTS_PATH` (usually `sqlite:///tasks.db` and `sqlite+aiosqlite:///checkpoints.sqlite`) via `.env` or Docker Compose environment variables if you want to store data in a custom location.

## Debugging Tips

- If an endpoint fails, check the logs for parsing errors (`_extract_json`)—Ollama occasionally wraps JSON in stray text, and the helper method tries to recover valid JSON.
- Whenever you edit `app.py`, rebuild the container (`docker-compose up -d --build`). For rapid iteration outside Docker, run `uvicorn app:app --reload` from a Python virtual environment.
- Keep an eye on the UI’s console/polling logs (`index.html` uses `console.error` for polling issues) so you know if the approval queue unexpectedly becomes empty or stuck.

Document new observability steps here (e.g., additional exports, alerting hooks) so the ops runbook stays current.
