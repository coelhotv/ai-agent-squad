# Project: Multi-Agent Product Squad

This repository orchestrates a LangGraph-powered **Coordinator** and a suite of specialist agents that intake ideas, research them, draft PRDs and user stories, and pause for Human-in-the-Loop approvals before shipping structured artifacts.

## Core Documentation

- **Overview:** `DOCS/overview.md` explains the vision, squad roles, and expected deliverables.
- **Architecture:** `DOCS/architecture.md` maps the hybrid Ollama + Docker stack, persistence layers, and API surface.
- **Setup & Prerequisites:** `DOCS/setup.md` walks through installing Ollama, Docker, pulling the model, and running the stack.
- **Workflow:** `DOCS/workflow.md` describes the step-by-step graph flow and how the FastAPI endpoints interact with the UI.
- **Roadmap:** `DOCS/roadmap.md` covers completed phases, the active Design Sprint, and future Engineering/QA/GTM milestones.
- **Operations:** `DOCS/operations.md` captures monitoring tips, dashboard/export guidance, and maintenance notes.

## Quick Start

1. Install [Ollama](https://ollama.com/) and Docker Desktop.
2. Run `ollama pull deepseek-r1:7b-qwen-distill-q4_K_M`.
3. Copy `.env.example` to `.env`, fill in any overrides (e.g., `PERPLEXITY_API_KEY`).
4. Start the app: `docker-compose up -d --build`.
5. Point your browser to `http://localhost:8000` for intake and `http://localhost:8000/tasks_dashboard` to watch the dashboard.

Refer to `DOCS/setup.md` for environment variables, dependency notes, and optional local development via a Python virtual environment.

## Workflow at a Glance

- `/start_task` creates a task, kicks off the research node, and persists the initial row in `tasks.db`.
- Research (Perplexity → DuckDuckGo) writes summaries, then the Product agent drafts PRDs and stories, each pausing via `respond_to_approval`.
- The UX agent now follows the Product steps, generating a Mermaid.js flow and Tailwind wireframe for approval before the work is marked `ready_for_engineering`, and the intake UI exposes “View Flow / View Wireframe” buttons that launch previews in new browser tabs.
- Every artifact panel includes an Edit action while the task is pending approval, allowing you to tweak the text/flow/wireframe, save it (persisted to `tasks.db`), and only then hit Approve so the downstream agents work with the human-updated version.
- The refreshed intake UI shows a real-time status pill, workflow stage cards, and the latest response message directly beneath the submission card to keep operators oriented.
- `tasks.html` polls `/tasks` every five seconds and offers an export button that hits `/tasks/export`.

See `DOCS/workflow.md` for the full narrative, including how checkpoints resume when humans approve.

## Next Steps

Continue expanding the roadmap from `DOCS/roadmap.md`, especially Phase 4+ workstreams (UX, Engineering, QA, GTM). Update `DOCS/operations.md` whenever you add new observability or export features so operations stay current.
