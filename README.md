# Multi-Agent Product Squad

This repo runs a LangGraph-based **Coordinator** that serializes a research- → product → UX → engineering sprint, pausing after each major output so a human reviewer can approve, edit, or reject the artifact before the next specialist continues. The FastAPI backend, SQLite task store, Ollama/Perplexity calls, and a single-page intake UI (index.html) work as a single loop that keeps every task resumable, traceable, and exportable.

## Highlights
- **Sequential agents:** Research → Product PRD → Product Stories → UX (flows + wireframes) → Architect spec with QA → Engineering code with QA → (ready for GTM). Each node interrupts for HITL approval and logs its status in `tasks.db`.
- **Artifact control:** All pending artifacts can be edited inline while the status is waiting (`pending_*`), edits persist to both the DB and the LangGraph checkpoint, and a rejection triggers a resubmit cycle that reruns that node with cleared downstream data.
- **UI & endpoints:** `index.html` shows workflow pills, a status badge, artifact collapsibles with edit overlays, Mermaid/wireframe preview buttons, approval buttons, and optimistic status updates backed by endpoints like `/start_task`, `/respond_to_approval`, `/update_artifact`, `/resubmit_step`, `/get_pending_approval`, `/tasks`, and `/tasks_dashboard`.
- **Resilience:** `tasks.db` tracks every artifact/QA report while `checkpoints.sqlite` (AsyncSqliteSaver) keeps the graph state live, so the app can resume a paused workflow on restart, and `/tasks/export` streams a CSV for auditing.
- **LLM routing:** Ollama drives PRD, stories, UX, spec, and code prompts (reasoning vs coding models), while the research node prefers Perplexity’s sonar-pro model and falls back to DuckDuckGo (`ddgs`) if the key is missing or the call fails.

## Artifacts & Deliverables
- Research summary with opportunities, risks, and references (Perplexity ⤑ DuckDuckGo fallback).
- One-page PRD and 2–3 user stories with acceptance criteria plus a tiny backlog.
- Mermaid.js user flow diagram and Tailwind/HTML wireframe preview.
- Architect-generated API spec (`schemas` + `endpoints`) plus its QA review.
- Engineering prototype (`main.py`-style code) plus QA output before final approval.
- (Future) GTM README/package notes once the workflow reaches the GTM node.

## APIs, UI, and Controls
- `POST /start_task`: kicks off a new task, writes a `Task` row, and runs the LangGraph until the first interruption.
- `POST /respond_to_approval`: approves/rejects regardless of node, resumes the graph, applies artifact overrides, updates statuses, and streams the new artifacts back to the UI.
- `POST /update_artifact`: persist edits while a task is pending so downstream nodes consume the updated content.
- `POST /resubmit_step`: reruns one node after a rejection by clearing downstream artifacts and invoking the node function directly.
- `GET /get_pending_approval`: polled on page load/polling to grab the next waiting task and lock the intake form until it is approved/rejected.
- `/tasks` and `/tasks_export` keep the dashboard in sync with `tasks.db`; `/tasks_dashboard` serves `tasks.html` and `/` serves `index.html`.
- The UI locks the form while a pending approval is active, renders status pills, shows approval prompts, and enables Flow/Wireframe previews in new tabs.

## Setup & Operations
1. Install [Ollama](https://ollama.com) and Docker Desktop (the app communicates with `http://host.docker.internal:11434`).
2. Pull the reasoning model (`deepseek-r1:8b-0528-qwen3-q4_K_M`) and coding model (`qwen2.5-coder:7b-instruct-q6_K`) so the reasoning/coding prompts stay aligned.
3. Copy `.env.example` → `.env`, set `PERPLEXITY_API_KEY` if available, and optionally override `OLLAMA_BASE_URL`, `OLLAMA_REASONING_MODEL`, `OLLAMA_CODING_MODEL`, and `DATABASE_URL`/`CHECKPOINTS_PATH`.
4. `docker-compose up -d --build` to launch the stack (it seeds `tasks.db` and `checkpoints.sqlite`, or creates the parent data directory if missing).
5. Use `index.html` for submitting ideas and approving artifacts, and `tasks.html`/`/tasks/export` for dashboards and exports.

## Documentation
- `DOCS/overview.md` – vision, mission, and cream-of-the-crop agent roles delivered today.
- `DOCS/architecture.md` – LangGraph nodes, FastAPI endpoints, Ollama + Perplexity routing, persistence, and QA mini-agents.
- `DOCS/workflow.md` – detailed flow of `/start_task`, approvals, artifact edits, resubmissions, and how the UI keeps everyone aligned.
- `DOCS/setup.md` – env vars, Ollama models, persistence considerations, and state-reset guidance.
- `DOCS/operations.md` – monitoring tips, logs to watch, and how to interact with pending approvals or exported data.
- `DOCS/components.md` – breakdown of backend, frontend, persistence, and tooling glue.
- `DOCS/project.md` + `DOCS/roadmap.md` – philosophical guardrails plus a phase-by-phase status of where the squad is headed.
