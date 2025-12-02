# AI Product Squad Components

## Backend & Graph

- **FastAPI (`app.py`):** Hosts `/start_task`, `/respond_to_approval`, `/update_artifact`, `/resubmit_step`, `/get_pending_approval`, `/tasks`, `/tasks/{task_id}`, `/tasks/export`, `/tasks_dashboard`, `/`, and `/status`. It wires request handling, logging, and the SQLite `Task` model (including the new `execution_mode` field) with the LangGraph graph.
- **LangGraph State Graph:** Seven nodes (`research`, `product_prd`, `product_stories`, `ux_design`, `engineering_spec`, `engineering`, `approved`) run sequentially, interrupting before each approval. `initialize_graph` compiles the graph with an `AsyncSqliteSaver` checkpointer so checkpoints live in `checkpoints.sqlite`.

## Persistence Layers

- **`tasks.db`:** Stores each artifact, QA review, filenames, statuses, pending content, and rejection metadata. SQLAlchemy ensures the schema evolves via `ensure_column`, so fields can be added without migrations.
- **`checkpoints.sqlite`:** Persisted LangGraph state used by `AsyncSqliteSaver`; it keeps the workflow resumable on restart, and `get_app_graph` throws a 503 if the graph is not yet initialized.
- **Artifact/Status Mapping:** Every node writes to a `pending_*` status (e.g., `pending_spec_approval`), `pending_approval_content`, and the `Task` row so the UI and CSV export show the same intermediate result.

## Agents & Tooling

- **Research Agent:** Uses `call_perplexity_json` with a strict schema for summaries, opportunities, risks, and references. If `PERPLEXITY_API_KEY` is absent or an exception occurs, it routes to `run_duckduckgo_research` (leveraging `DDGS.text`/`news`).
- **Product Agent:** `generate_prd_document` and `generate_user_stories` send reasoning prompts to the appropriate Ollama model and format the structured JSON into readable text.
- **UX Agent:** `generate_user_flow_diagram` returns Mermaid syntax and `generate_wireframe_html` returns Tailwind-style markup; both prompts enforce schema validation to keep the output predictable.
- **Architect + Developer + QA Agents:** `generate_engineering_spec` now builds `detailed_steps`, enforces acceptance criteria coverage with `ac_refs`, and auto-injects placeholder schemas/endpoints whenever the reasoning model forgets an AC, exposing any warnings along with the contract. `run_spec_qa_review` consumes the full spec (warnings included), fails whenever findings exist, and appends a checklist that reminds humans about pending ACs or demo constraints. `generate_engineering_code` receives those warnings, produces a detailed implementation plan with generic event/counter/notification guidance, and the final `run_engineering_qa_review` enforces the same AC coverage plus the demo-only keyword guardrails before it marks the code passable.
- **LLM Routing Helpers:** `call_ollama_json` selects reasoning vs coding models, handles retries/timeouts, and logs token usage. `_extract_json` tolerates stray text around JSON payloads.

## Frontend

- **`index.html`:** Presents the hero intake card, workflow timeline, approval banner, artifact collapsibles with edit overlays, Mermaid/wireframe preview buttons (open in new tabs), optimistic status updates, queue refreshing, and rejection/resubmission handling. The Manual/Semi-auto toggle posts `execution_mode`, and a task-monitor loop (polling `/tasks/{task_id}`) streams live status/artifact updates while semi-auto advances early stages.
- **`tasks.html`:** Displays the entire `tasks.db` table, polls `/tasks`, and shows status badges/export controls; `/tasks_dashboard` serves this file via `FileResponse`.
- **Artifact Editing:** While a task is pending, each artifact card enables an Edit button that opens an overlay. The overlay submits `POST /update_artifact` with the new content, which immediately updates both the UI and LangGraph state so downstream agents see the edits.

## Monitoring & Logging

- Python `logging` records environment status (database URL, checkpoints path, Ollama host/models, Perplexity key presence) and per-node activities; SQLAlchemy logs are reduced to WARNING to avoid dashboard noise.
- Docker Compose logs (`docker-compose logs -f app_service`) show each node’s progress and QA verdicts. The UI polls `/get_pending_approval` and `/tasks`, so any failure in the backend quickly surfaces in the browser console.
- The `ASYNC` graph ensures `await graph.ainvoke` and `graph.aget_state` keep the UI and DB synchronized even if approvals arrive slowly.

See `DOCS/architecture.md`, `DOCS/workflow.md`, and `README.md` for deeper narratives on how these components interact.
