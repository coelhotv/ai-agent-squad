# Roadmap

Each phase represents a major milestone in the Multi-Agent Product Squad. Status notes keep the team aligned.

## Completed Phases

### Phase 1 – The Stack (✅)

- Hybrid architecture: Dockerized FastAPI + LangGraph, native Ollama host.
- `docker-compose.yml`, `Dockerfile`, and `requirements.txt` created.
- Ollama model switched to `deepseek-r1:8b-llama-distill-q4_K_M`.
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

## In Progress

### Phase 4 – Design Sprint (🟡)

Objective: Add Product/UX agents that deliver PRDs, user stories, and design artifacts.

Current work:
- Product agent now drafts PRDs and user stories with two HITL approvals.
- Dashboard refreshed with elastic table layout and export CSV.

Next steps:
1. Define UX prompts for Mermaid.js flows + HTML/Tailwind wireframes.
2. Create UX node that pauses for HITL approval before handing off to Engineering.
3. Capture wireframes/flows in `tasks.db` and checkpoint state.

Acceptance:
- UX artifacts generated on first approval.
- Human approval UI reflects new artifacts.
- Checkpointer stores UX progress for reliable resumes.

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
