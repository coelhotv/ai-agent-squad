# Roadmap

Each phase represents a major milestone in the Multi-Agent Product Squad. Status notes keep the team aligned.

## Completed Phases

### Phase 1 – The Stack (✅)

- Hybrid architecture: Dockerized FastAPI + LangGraph, native Ollama host.
- `docker-compose.yml`, `Dockerfile`, and `requirements.txt` created.
- Ollama model switched to `deepseek-r1:8b-0528-qwen3-q4_K_M`, and we now pair it with `qwen2.5-coder:7b-instruct-q6_K` for coding-intensive specialists.
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

Now we can let humans edit every artifact before approving it so they can improve the research/PRD/stories/UX work instead of only gating it.
- Add edit controls next to each artifact that are active only while the corresponding `pending_*` status is waiting.
- Persist edits to `tasks.db` and pass the new text back into the graph state (or reload it from the DB) so downstream agents consume the human updates.
- Allow the preview buttons to render the edited flow/wireframe content, and lock the fields once the task advances past approval/rejection.
- Included a re-submit flow for artifacts that are rejected. Now HITL could redo the last agent flow to create a new version of its artifact.

### Phase 5 – Build Sprint (✅)

- Engineering runs as a bundle: spec (schemas + endpoints), code generation, and QA review in one HITL step.
- Approvers see the spec, code, and QA output together at `pending_engineering_bundle_approval`; edits/resubmits rerun the bundle.
- New artifacts (`engineering_spec`, `engineering_code`, `engineering_qa`) are rendered in the UI and exported via `/tasks/export`.

## In Progress

- None

## Future Phases

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
