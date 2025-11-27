# Roadmap

Each phase documents a major milestone for the Multi-Agent Product Squad. Use this doc to track progress, celebrate completions, and signal what remains.

## Completed Phases

### Phase 1 – Stack Foundation (✅)
- Hybrid architecture: Dockerized FastAPI/LangGraph runs inside `app_service` while Ollama (hosted) handles LLM inference.
- Persistent storage split between `tasks.db` (high-level artifacts/status) and `checkpoints.sqlite` (LangGraph’s `AsyncSqliteSaver` state).
- Initial endpoints (`/start_task`, `/tasks`, `/respond_to_approval`, `/get_pending_approval`) and a barebones UI existed to validate the serialized workflow.

### Phase 2 – HITL Control & Coach UI (✅)
- The intake page (`index.html`) now renders workflow pills, status badges, approval prompts, response messages, and artifact cards with edit overlays.
- `/tasks_dashboard` (`tasks.html`) and `/tasks/export` provide auditing/export capabilities while polling keeps the UI fresh.
- Approval logic was refactored so each node sets `pending_*` statuses, stores `pending_approval_content`, and waits for `/respond_to_approval`.

### Phase 3 – Research + Product (✅)
- Research agent calls Perplexity `sonar-pro` and falls back to DuckDuckGo when necessary; the Research node writes structured findings into both the graph and DB.
- Product agent now writes a concise PRD and user stories, each pausing for a separate approval window. Artifact edits persist back into LangGraph state.

### Phase 4 – UX + Collaborative Approvals (✅)
- UX agent builds Mermaid flows and Tailwind wireframes; the UI preview buttons open those artifacts in new tabs, and exports include `user_flow_diagram`/`wireframe_html`.
- Humans can edit any pending artifact (research, PRD, stories, flow, wireframe) via `/update_artifact`; overrides sync to the checkpoint so downstream agents honor the edits.
- A resubmit flow (`/resubmit_step`) reruns the rejected node after clearing downstream artifacts, giving humans a quick path to regenerate specific deliverables.

### Phase 5 – Engineering Bundle + QA (✅)
- Architect reasoning + contract generation now emit structured `schemas`/`endpoints`, followed by a spec QA review stored in `engineering_spec_qa`.
- Developer agent implements the spec as code, then runs an engineering QA review (`engineering_qa`) before pausing at `pending_code_approval`.
- The UI shows spec/code/QA artifacts, and CSV exports include all engineering outputs.

### Phase 5.5 – Semi-Auto Mode (✅)
- Added Manual/Semi-auto toggle on the intake form plus backend `execution_mode` tracking.
- Built `auto_advance_until_spec` and the `/tasks/{task_id}` snapshot endpoint so Research → PRD → Stories → UX auto-run while the UI streams live status/artifact updates.
- `/get_pending_approval` now skips auto-advancing tasks until they pause at Spec Reasoning, keeping HITL focused on later stages.

## In Progress

- None. The stack currently pauses at `pending_code_approval` before a future GTM node marks tasks `ready_for_gtm`.

## Future Phases

### Phase 6 – GTM & Ship (🟢)
- Add a GTM agent that synthesizes the artifacts into a polished README/package deliverable plus launch notes.
- Extend the graph to mark `ready_for_gtm` (or `completed`) once GTM work is approved, and reflect that final status in both the UI and `/tasks` exports.
- Document any new monitoring, onboarding, or release steps inside `DOCS/operations.md`.

As phases evolve, adjust this roadmap along with `DOCS/overview.md`, `DOCS/architecture.md`, and `README.md` so contributors always have a consistent narrative.
