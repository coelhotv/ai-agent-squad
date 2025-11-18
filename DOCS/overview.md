# Overview

## Vision

We are building a **Multi-Agent Product Squad** that can intake early-stage ideas, research them, draft specifications, and hand off actionable artifacts—while keeping a person in the loop for every key decision. The squad is orchestrated by a LangGraph-based **Coordinator** that serializes specialist work so the overall process remains predictable and auditable.

## Mission

Deliver a full “scoped shipped package” for every validated idea, including a research briefing, concise PRD, user stories, Mermaid.js flows, wireframes, a prototype, and a GTM-ready README—without the human ever losing visibility into progress.

## Core Goals

1. Keep the flow **serialized**: only one agent runs at a time, and each specialist returns structured output for the next agent to consume.
2. Guarantee **persistence**: a SQLite `tasks.db` tracks high-level task status, and `checkpoints.sqlite` lets LangGraph pause and resume without losing internal state.
3. Maintain **HITL control**: every major handoff pauses for human review via the FastAPI frontend (`index.html`, `/tasks_dashboard`).

## Squad Roles

- **Coordinator:** Manages the LangGraph state graph, checkpoints, and transitions between nodes. It also exposes FastAPI endpoints like `/start_task`, `/respond_to_approval`, and `/tasks` for the UI.
- **Human (You):** Reviews research, PRDs, stories, and code; approves/rejects or edits outputs using the web UI.
- **Research Agent:** Calls Perplexity (`sonar-pro`) when `PERPLEXITY_API_KEY` is available, otherwise falls back to DuckDuckGo (`ddgs`) to provide research summaries with opportunities, risks, and references.
- **Product Agent:** Drafts PRDs (exec summary, opportunity, needs, scope, success metrics) and user stories with acceptance criteria before passing control to UX.
- **UX/Designer Agent:** Produces Mermaid user flows and HTML/Tailwind wireframes once Product work is approved, pausing for HITL sign-off before engineering picks it up.
- **Engineering Agent (forthcoming):** Will generate working prototypes and backend logic.
- **QA Agent (forthcoming):** Will review generated code for bugs, regressions, and quality issues.
- **GTM Agent (forthcoming):** Will write final README.md and packaging notes before marking the task as shipped.

## Output Checklist

Every completed task should aim to deliver:

- Validated research notes (Perplexity or DuckDuckGo) stored in `tasks.db`.
- A succinct PRD with clear scope and success criteria.
- 3–4 user stories with acceptance criteria and a small backlog.
- Mermaid.js flows + HTML/Tailwind wireframes (Phase 4+).
- Prototype code (Python or other single-file artifact).
- QA-reviewed logic and GTM README (final phases).

Link back to the living architecture docs in `DOCS/architecture.md` and workflow notes in `DOCS/workflow.md` to understand how each output is produced.
