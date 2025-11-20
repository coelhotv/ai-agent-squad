# Roadmap

Each phase represents a major milestone in the Multi-Agent Product Squad. Status notes keep the team aligned.

## Completed Phases

### Phase 1 – The Stack (✅)

- Hybrid architecture: Dockerized FastAPI + LangGraph, native Ollama host.
- `docker-compose.yml`, `Dockerfile`, and `requirements.txt` created.
- Ollama model switched to `deepseek-r1:14b-qwen-distill-q4_K_M`.
- SQLite persistence established (`tasks.db`, `checkpoints.sqlite`).
- Hello-world flow validated (logging, connectivity, vscode/.venv setup).

### Phase 2 – Core Loop & HITL UI (✅)

- FastAPI endpoints: `/start_task`, `/get_pending_approval`, `/respond_to_approval`, `/tasks`.
- LangGraph workflow reworked around `AsyncSqliteSaver` for reliable pause/resume.
- UI for intake (`index.html`) and dashboard (`tasks.html`) built.
- `tasks.db`/`checkpoints.sqlite` separation enforced to keep high-level UI state apart from graph internals.
- `/tasks_dashboard` added for easy auditing.

### Phase 3 – Research Agent (✅)

- Research node calls Perplexity (`sonar-pro`) with structured JSON; fallback to DuckDuckGo.
- Research summary stored in both DB and LangGraph state.
- UI surfaces pending research and requires HITL approval before continuing.

### Phase 4 – Design Sprint (✅)

- Product agent drafts PRDs and user stories with two HITL approvals.
- UX/Designer agent generates Mermaid flows + Tailwind wireframes via structured `/api/generate` calls, keeps them in sync, and surfaces them with preview buttons that open new tabs.
- Artifacts are persisted (`user_flow_diagram`, `wireframe_html`) and exposed via `/tasks_dashboard` and CSV export.
- Main intake UI now highlights workflow stages, status messages, and artifact controls so operators can stay on one screen.

### Phase 4.5 – Collaborative Approvals (✅)

Before moving to Phase 5 we will let humans edit every artifact before approving it so they can improve the research/PRD/stories/UX work instead of only gating it. Planned work:
1. Add edit controls next to each artifact that are active only while the corresponding `pending_*` status is waiting.
2. Persist edits to `tasks.db` and pass the new text back into the graph state (or reload it from the DB) so downstream agents consume the human updates.
3. Allow the preview buttons to render the edited flow/wireframe content, and lock the fields once the task advances past approval/rejection.
4. Once this collaboration layer is stable, proceed to Phase 5 (Engineering + QA).

## In Progress

- None

## Future Phases

### Phase 5 – Build Sprint (🟠)

Objective: Add Engineering and QA agents.

Planned work:
- Engineering agent produces a single-file prototype and supporting backend logic.
- QA agent reviews the generated code and flags regressions or missing tests.

Success Criteria:
- Code artifacts stored in task record or surfaced via dashboard.
- QA approvals gate shipping to next phase.
- Logging captures QA decisions.

### Phase 6 – Ship (🟢)

Objective: Add GTM agent and finalize the deliverable package.

Planned work:
- GTM agent writes a polished `README.md`.
- Task output (research, PRD, stories, flows, prototype, README) is collected under `dist/` or similar folder.
- Workflows include a final approval for the GTM article.

Success Criteria:
- Full artifact bundle is easy to download or examine.
- Dashboard highlights shipped tasks separately.
- Hand-off documentation updated (`DOCS/operations.md`) with current ops steps.
