# Workflow & Approval Loop

## End-to-End Flow

The squad follows a serialized, checkpointed sequence. Each node in the LangGraph graph runs once, outputs a structured result, and then pauses for a human approval before continuing. FastAPI endpoints and the UI keep you in sync with the workflow.

### 1. Submit an Idea

Navigate to `http://localhost:8000` (served by `index.html`), enter a product idea, and click **"Start Task"**. The browser calls `POST /start_task`, which:

1. Generates a UUID `task_id`.
2. Saves a new row to `tasks.db` with `status="starting"`.
3. Triggers the LangGraph workflow with `thread_id=task_id` via `AsyncSqliteSaver`.

The initial task record appears immediately on `tasks_dashboard` (`tasks.html`), which polls `GET /tasks` every five seconds.

### 2. Research Node

The Research node executes first. It tries to call Perplexity’s `sonar-pro` model using `PERPLEXITY_API_KEY` to get structured JSON (summary, opportunities, risks, references). If that fails, it falls back to DuckDuckGo via `ddgs`.

Once research text is produced:

- The LangGraph state is updated (`status="pending_research_approval"`).
- Research text is saved to the task row.
- A human receives the info in the UI.

The state pauses and waits for manual approval.

### 3. Product Agent (PRD & User Stories)

After Research is approved via the UI, `POST /respond_to_approval` resumes the graph:

1. Product agent drafts a concise PRD (executive summary, market opportunity, customer needs, scope, success metrics).
2. It pauses again with `status="pending_prd_approval"` to show the PRD in the UI.
3. After PRD approval, the `respond_to_approval` endpoint runs the Product node again, which now produces user stories + acceptance criteria.
4. The state updates to `pending_story_approval`.

Each pause is captured in `checkpoints.sqlite`, so crashes/resets do not lose context.

### 4. Watching the Dashboard

`tasks.html` renders the full table of tasks with elastic columns and proper status badges. The manual refresh button hits `GET /tasks`, while auto polling keeps data fresh. Export everything at once with `GET /tasks/export` (via the **Export CSV** button).

### 5. Next Specialist Phases

Currently `status="ready_for_ux"` marks the handoff to the upcoming UX agent (Mermaid flows + Tailwind wireframes). Future nodes (Engineering, QA, GTM) will keep the same pattern:

- Produce structured output.
- Save it to the task record.
- Pause for human approval via `respond_to_approval`.
- Continue once the human gives the green light.

The `tasks.db` row and `checkpoints.sqlite` resume state so, at any time, you can reopen the dashboard and see where each task stands.
