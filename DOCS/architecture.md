# AI Product Squad Architecture

This document maps how FastAPI, LangGraph, SQLite, and the Ollama/Perplexity tooling work together to deliver the squad’s serialized workflow.

## Core Components

- **FastAPI backend (`app.py`):** Exposes the public API, manages `tasks.db`, coordinates checkpoint updates, and serves static UI assets (`index.html`, `tasks.html`). It loads configuration via `app_settings.get_settings()`, logs environment details, and ensures the SQLite schema evolves through `ensure_column`.
- **LangGraph + AsyncSqliteSaver:** The `workflow` StateGraph defines nodes for each specialist (research → PRD → stories → UX → engineering spec → engineering code → approved). The graph compiles once on startup and uses `AsyncSqliteSaver.from_conn_string(settings.checkpoints_path)` so every run writes into `checkpoints.sqlite`; it interrupts before each node listed in `interrupt_before`, enabling human approvals.
- **SQLite & Task Model:** `tasks.db` stores the `Task` ORM model with every artifact field (`research_summary`, `prd_summary`, etc.), status metadata (`status`, `pending_approval_content`, `last_rejected_step`, etc.), and ensures the UI can query/persist all artifacts.
- **Vanilla JS UI:** `index.html` drives the HITL experience: workflow cards, approval buttons, artifact collapsibles, edit overlays, Mermaid/wireframe previews, optimistic statuses, and a queue refresher. `/tasks.html` (served via `/tasks_dashboard`) polls `/tasks`.

## LangGraph Workflow

1. **Start task (`start_task` node):** Creates a UUID, persists a new `Task` record (`status="starting"`), defines `AgentState`, and invokes `graph.ainvoke` until the first interruption. The graph’s `thread_id` equals the `task_id`.
2. **Node sequence:** Research → PRD → Stories → UX → Engineering spec → Engineering code → Approved. Each node generates artifacts, updates `state.status` to a `pending_*` flag, and writes a user-facing `pending_approval_content` description before returning. The `approved` node marks the status as `completed` and clears the pending text.
3. **Interrupts:** The graph interrupts before the first five nodes plus engineering checkpoints (`interrupt_before` list). After each interruption, `start_task`/`respond_to_approval` fetch `graph.aget_state` to sync `tasks.db` with the in-memory state.

## Persistence & State

- **`tasks.db`:** Default path from settings (usually `sqlite:///tasks.db`). The `Task` model mirrors `AgentState` fields, so artifacts and QA reports remain queryable for dashboards, exports, and overlay edits.
- **`checkpoints.sqlite`:** Lives next to the data directory; LangGraph stores every state, allowing the system to resume work after a restart without replaying earlier nodes.
- **Status metadata:** Pending states (`pending_research_approval`, `pending_prd_approval`, etc.) map back to node names via `status_to_step`, enabling `/respond_to_approval` to know which node was waiting. `clear_artifacts_for_step` wipes downstream artifacts before resubmitting.
- **DB evolutions:** `ensure_column` adds columns on startup so optional artifacts can be added safely without migrations.

## LLM Routing

- **Perplexity Research:** `run_research_query` prefers `call_perplexity_json` with a strict schema (summary, opportunities, risks, references). If `PERPLEXITY_API_KEY` is missing or the call fails, it runs `run_duckduckgo_research` using `DDGS.text/news`.
- **Ollama Agents:** `call_ollama_json` selects either the reasoning model (`OLLAMA_REASONING_MODEL` or fallback) or the coding model via `_get_ollama_model`. Every agent (PRD, stories, UX flow, wireframe, architect reasoning, contract, code, QA reviews) sends structured JSON schemas so the responses can be safely parsed.
- **Reasoning vs coding split:** The engineering spec node uses a reasoning prompt to generate `reasoning_steps`, then the coding prompt to emit the `schemas`/`endpoints` contract; `generate_engineering_code` consumes that contract and produces the final prototype.
- **Acceptance criteria enforcement:** The spec pipeline now tracks which ACs appear via `ac_refs` arrays. If any AC is missing, `append_ac_placeholder` injects a generic schema/endpoint pairing so the contract always mentions the obligation, and the warnings list notes the injection. These warnings travel with the spec into the coding and QA stages, guiding them to simulate the expected counters, aggregates, and notification mocks without hardcoded domain references.

## QA + Human-in-the-Loop

- `/respond_to_approval`: Validates the pending status, supports overrides via `artifactOverrides`, marks the status as `processing`, resumes the graph with `graph.ainvoke`, and updates `tasks.db` with the returned artifacts.
- `/update_artifact`: Allows inline edits while a task is pending; it writes the new text to both `tasks.db` and the LangGraph checkpoint so downstream agents see the human spin.
- `/resubmit_step`: Clears downstream artifacts, rebuilds an `AgentState` from the DB, moves `status` to the requested `pending_*` phase, and re-invokes the relevant node function for quick reruns after rejection.
- `/tasks/export`: Streams a CSV with every artifact column plus `pending_approval_content` for audits.
- UI resilience: `index.html` polls `/get_pending_approval`, locks submissions when work is in flight, renders resubmit banners when statuses hit `rejected`, and only reenables editing once the task returns to a pending state.
- **QA enforcement:** `run_spec_qa_review` now fails whenever the parsed findings list is non-empty and appends a checklist reminding reviewers about missing ACs or demo constraints; this stops the workflow from pretending a spec is “pass” when it still lacks required artifacts. Likewise, `run_engineering_qa_review` compares the runtime code against the spec AC references, watches for the banned keywords we track in `DEMO_BANNED_KEYWORDS`, and highlights any simulated-notification gaps so real push reminders never get forgotten.

Together, these layers keep the Multi-Agent Product Squad fast, auditable, and human-centered.
