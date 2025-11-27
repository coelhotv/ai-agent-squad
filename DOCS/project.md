# Project: Multi-Agent Product Squad

## Vision & Stack

We are orchestrating a **Multi-Agent Product Squad** that turns a single idea into a structured deliverable bundle (research, PRD, stories, UX flows/wireframes, spec, prototype code) by running one agent at a time and pausing for human review before continuing. Reviewers can now choose between Manual and Semi-auto submission modes; semi-auto auto-approves Research → PRD → Stories → UX before pausing at Spec Reasoning, with the UI streaming intermediate statuses/artifacts.

- **Coordinator + LangGraph:** FastAPI (`app.py`) defines the graph, checkpoints with `AsyncSqliteSaver`, and exposes the endpoints the UI uses to publish statuses, approvals, and artifact edits.
- **LLM Hosts:** Ollama (`deepseek` for reasoning/QA, `qwen2.5-coder` for UX/engineering) runs outside Docker; Perplexity’s `sonar-pro` handles structured research when `PERPLEXITY_API_KEY` is available, with DuckDuckGo fallback otherwise.
- **Persistence:** `tasks.db` stores every artifact/QA record while `checkpoints.sqlite` lets the workflow pause and resume; Docker Compose mounts them in the `app_data` volume (`/data` inside the container).
- **Frontend:** `index.html` paints the workflow timeline, artifact cards, edit overlay, and preview buttons; `tasks.html` renders the full table plus a CSV export button.

## Agent Roles

- **Research Agent:** Hits Perplexity/DuckDuckGo, formats the summary, and pauses with `pending_research_approval`.
- **Product Agent:** Drafts the PRD, then user stories with acceptance criteria, pausing between phases to let a human review and edit.
- **UX/Designer Agent:** Generates Mermaid flows and Tailwind wireframes that the UI can preview in new tabs. Both artifacts can be edited while the status is `pending_ux_approval`.
- **Architect Agent:** Runs a reasoning prompt to describe schemas/endpoints, then a coding prompt to produce the contract JSON, followed immediately by a QA review (`run_spec_qa_review`). The tail end of this stage pauses at `pending_spec_approval`.
- **Engineering Agent:** Consumes the spec to write a single FastAPI prototype, runs `run_engineering_qa_review`, and pauses at `pending_code_approval` so humans can read the code + QA summary.
- **QA Helpers:** Both the spec and code are sent through Ollama QA prompts, and their findings are captured in `engineering_spec_qa` and `engineering_qa`.
- **GTM Agent (future):** Will compile the artifacts into a README/package notes and mark tasks ready for GTM.
- **Human Reviewer (You):** Approves artifacts, can edit any pending artifact via `/update_artifact`, and can resubmit the last rejected step with `/resubmit_step` (e.g., rerun the research node if the research summary needs a rewrite).

## Project Phases

- **Phase 1 – Hybrid Stack (✔️):** Dockerized FastAPI + LangGraph with Ollama on host, `docker-compose.yml`, `Dockerfile`, `requirements.txt`, and the checkpoint architecture.
- **Phase 2 – Coordinator & HITL UI (✔️):** `POST /start_task`, `/get_pending_approval`, `/respond_to_approval`; `index.html` and `tasks.html`; `tasks.db` vs `checkpoints.sqlite` separation.
- **Phase 3 – Research Agent (✔️):** `sonar-pro` research with DuckDuckGo fallback, stored summaries, and human approval before PRD generation.
- **Phase 4 – Product + UX Sprint (✔️):** PRD + user stories, Mermaid flow + wireframe generation, UI preview buttons, and artifact Edit overlays. The UI now locks the form when an approval is pending, and exports include UX artifacts.
- **Phase 4.5 – Collaborative Approvals (✔️):** Humans can edit pending artifacts, the backend syncs overrides into LangGraph, and resubmissions rerun only the rejected node while clearing downstream fields.
- **Phase 5 – Engineering Bundle (✔️):** Architect spec → QA → code → QA two-step, with `pending_spec_approval` and `pending_code_approval`, artifact cards for specs/code/QA, and CSV exports capturing every field.
- **Phase 5.5 – Semi-auto Onramp (✔️):** Manual/Semi-auto toggle, auto-advance loop (Research → PRD → Stories → UX), `/tasks/{task_id}` monitor endpoint, and real-time UI updates while early stages fly by.
- **Phase 6 – GTM & Ship (🟢 soon):** Future work aims to add a GTM agent that writes the README/package deliverables and marks tasks as `ready_for_gtm`/`completed`.

This living doc stays aligned with `DOCS/workflow.md`, `DOCS/architecture.md`, and `README.md`; any shift in the graph, endpoints, or UI should be noted here for the broader squad to absorb.
