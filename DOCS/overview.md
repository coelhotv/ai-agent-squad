# Overview

## Vision

We are building a **Multi-Agent Product Squad** that intakes early-stage ideas, researches them, drafts product artifacts, designs UX, and delivers prototype code—with each specialist pausing for a human approval before the next stage begins. LangGraph orchestrates the flow so every task is serialized, auditable, timeout-aware, and resumable.

## Mission

Ship a scoped, human-reviewed deliverable bundle for every validated idea: research intelligence, a focused PRD, actionable user stories, UX flows/wireframes, an API spec, prototype code, and eventually GTM packaging—all while keeping a person in the loop for every checkpoint and edit.

## What Falls Out of the Graph

- Research summary (Perplexity `sonar-pro` JSON → human-readable text, with DuckDuckGo fallback).
- PRD with executive summary, opportunity, target customer, scope, and success criteria.
- User stories with 2–3 acceptance criteria plus a tiny backlog.
- Mermaid.js user flow diagram and Tailwind/HTML wireframe that preview in the UI.
- Architect-generated API spec (`schemas` + `endpoints`) plus a QA review of its completeness.
- Engineering prototype code (`main.py`-style FastAPI) plus QA commentary before final approval.
- Human edits captured on any pending artifact and synchronized back into the LangGraph checkpoint.
- Status metadata (`status`, `pending_approval_content`, `last_rejected_step`, `last_rejected_at`) persisted in `tasks.db` for dashboards and exports.

## Squad Roles

- **Coordinator (LangGraph + FastAPI):** Builds the graph, checkpoints, and exposes `/start_task`, `/respond_to_approval`, `/update_artifact`, `/resubmit_step`, `/tasks`, `/tasks/export`, `/get_pending_approval`, `/tasks_dashboard`, and static roots; it also manages the SQLite schema evolution (`ensure_column`) and logging.
- **Human (You):** Reviews every artifact on `index.html`, approves/rejects via `/respond_to_approval`, edits via `/update_artifact`, and can resubmit a rejected step through `/resubmit_step`.
- **Research Agent:** Calls Perplexity `sonar-pro` (with `PERPLEXITY_API_KEY`) for structured findings and falls back to DuckDuckGo via `ddgs` when the key, network, or LLM is unavailable.
- **Product Agent:** Writes a digestible PRD, then follows up with MVP-bound user stories (two high-value stories + acceptance criteria).
- **UX/Designer Agent:** Generates a Mermaid.js flow and uses the same context to craft a low-fidelity Tailwind/HTML wireframe preview.
- **Architect Agent:** Runs a reasoning Ollama prompt (reasoning model) to outline schemas/endpoints, then asks the coding model to emit the contract JSON.
- **Engineering Agent:** Implements the contract as a single-file FastAPI prototype, then triggers a QA review of the spec and code before human approval.
- **QA Agents:** Immediately review the architect spec and the generated code for demo-level completeness; their findings are saved as artifacts.
- **GTM Agent (future):** Will translate the finished artifacts into a README/package for launch and mark the task as `ready_for_gtm`/`completed`.

## Workflow Controls

Each node in the graph runs once, writes its artifact into `tasks.db`, and interrupts with a `pending_*` status (`pending_research_approval`, `pending_prd_approval`, `pending_story_approval`, `pending_ux_approval`, `pending_spec_approval`, `pending_code_approval`). `index.html` polls `/get_pending_approval`, locks the submission form, renders the pending artifact(s), and shows approval/reject buttons. Humans can edit any artifact while its status is pending—the edit sends `/update_artifact`, which writes back to the graph checkpoint so downstream agents see the updated copy. If a task is rejected, the UI surfaces a resubmit banner; calling `/resubmit_step` reruns only that node and clears downstream fields so the graph restarts cleanly.

Link back to `DOCS/architecture.md`, `DOCS/workflow.md`, and `README.md` for implementation specifics and quick-start instructions.
