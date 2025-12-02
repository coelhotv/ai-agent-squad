# Workflow & Approval Loop

This document tracks every turn of the graph from `POST /start_task` through approvals, artifact edits, and eventual completion.

## 1. Submit an Idea

1. The user loads `index.html` (served at `/`) and submits a product idea via `POST /start_task`. The form includes a Manual/Semi-auto toggle; the backend stores the choice as `execution_mode`.
2. FastAPI creates a `Task` record (`status="starting"`) in `tasks.db`, defines the initial `AgentState`, and invokes the LangGraph graph with `thread_id=task_id`.
3. The graph runs the `research` node, writes its artifact fields, updates `status="pending_research_approval"`, and interrupts before moving on. The backend then syncs `tasks.db` with the LangGraph state, returning the task to the UI with the research summary and approval prompt.
4. If another task is still waiting for approval, `index.html`’s `checkPendingApprovalOnLoad`/polling prevents new submissions until the pending work resolves.

## 2. Manual vs. Semi-Auto Progression

- **Manual:** Every node runs once, sets a `pending_*` status, and waits for HITL approval via `/respond_to_approval`. The UI locks the intake form until the pending approval is addressed.
- **Semi-auto:** The backend spawns `auto_advance_until_spec`, which loops Research → PRD → Stories → UX without a human click. `index.html` polls `/tasks/{task_id}` every ~2s to refresh the status card, pills, and artifacts as each stage completes. `/get_pending_approval` intentionally skips tasks still auto-advancing, so HITL users are only interrupted when Spec Reasoning starts.

## 3. Node-by-Node Progression (after auto phase)

- **Research** (`pending_research_approval`): Calls Perplexity `sonar-pro` (or DuckDuckGo) to produce summaries, opportunities, risks, and references. UI shows the research accordion and an Edit button while waiting for approval.
- **Product PRD** (`pending_prd_approval`): The next run drafts a one-page PRD, updates the task, and pauses for human review.
- **User Stories** (`pending_story_approval`): The product node returns again with stories & acceptance criteria; the UI surfaces them with an edit option before moving to UX.
- **UX Design** (`pending_ux_approval`): The UX agent emits a Mermaid user flow and a Tailwind/HTML wireframe. Buttons labeled “View Flow Diagram” and “View Wireframe” enable full-screen previews. While pending, the UI enables their Edit overlay inputs as well.
- **Engineering Spec** (`pending_spec_approval`): The architect prompts Ollama reasoning/coding models to produce schemas & endpoints, runs a QA review (`run_spec_qa_review`), and pauses so humans can read the spec + QA output.
- **Engineering Code** (`pending_code_approval`): The developer model implements the contract in a single-file FastAPI prototype, runs `run_engineering_qa_review`, and pauses for review.
- **Frontend Plan** (`pending_frontend_plan_approval`): LLM proposes how to surface the API in a React/Vite UI using idea + stories + wireframe + spec/code inputs.
- **Frontend Code** (`pending_frontend_code_approval`): LLM emits a React/Vite bundle and we attach validator warnings into a JSON blob `{bundle, warnings}`; HITL sees both, and the fallback template is used only if the bundle is empty.
- **DevOps Plan** (`pending_devops_plan_approval`): Human-readable deployment steps (no Docker syntax) describing a single-container build/run for API + frontend.
- **DevOps Test** (`pending_devops_test_approval`): LLM outputs Dockerfile + smoke tests; we store `{dockerfile, smoke, warnings}` so HITL can review the output even if warnings exist. Fallback is used only when the LLM returns nothing.

Each approval returns to `/respond_to_approval`, which logs the transition and, if approved, resumes the graph. The UI optimistically advances the workflow stage while the backend completes the next node. If you reject, the task status becomes `rejected`, records `last_rejected_step`, timestamps `last_rejected_at`, and exposes the resubmit banner in the UI.

## 4. Human Editing & Overrides

- While a task status is in `PENDING_STATUSES` (see `app.py`), every artifact card shows an Edit button. The overlay sends `POST /update_artifact`, which writes to both `tasks.db` and the LangGraph checkpoint so the next agent consumes the updated content. Warning-carrying artifacts (frontend_code, devops_compose) stay as JSON so HITL can see the LLM output plus validator findings.
- Approvals can include overrides (the UI persists `artifactOverrides`) that are sent with `/respond_to_approval` so the backend updates both the DB and the graph state before continuing.
- If a task is rejected, the UI renders a resubmit banner with `resubmitStep(task_id, last_rejected_step)`; `/resubmit_step` clears downstream fields via `clear_artifacts_for_step` and runs the node function directly so only that stage regenerates.

## 5. Pending Queue & Dashboard

- `/get_pending_approval` returns the oldest pending task that is not currently auto-advancing; semi-auto tasks reappear once they pause at Spec Reasoning.
- `/tasks/{task_id}` streams live snapshots so the UI can show which auto stage is active and progressively render artifacts as they finish.
- `/tasks` feeds the `/tasks_dashboard` table, and `/tasks/export` streams `tasks_export.csv` with every artifact column plus `pending_approval_content`, `last_rejected_*`, and other metadata for auditing.
- The dashboard polls every five seconds, so the latest statuses (processing, pending, ready_for_gtm, completed, rejected) remain visible even when multiple tasks flow through the queue.

## 6. Persistence & Recovery

- `checkpoints.sqlite` (AsyncSqliteSaver) records every restartable state so, after a crash, `start_task`/`respond_to_approval` can reload the same thread and resume from the last interrupt.
- `tasks.db` mirrors the human-visible data so both the UI and `/tasks_export` can render artifacts even if the graph is mid-run.
- The intake UI blocks new submissions until `currentTaskId` settles to `null`, ensuring humans finish one approval before starting another idea.

Use `DOCS/architecture.md` for the backend interplay, `DOCS/operations.md` for monitoring/output, and `DOCS/setup.md` when configuring Ollama/Perplexity.
