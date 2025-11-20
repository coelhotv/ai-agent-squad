# Project: Multi-Agent Product Squad

## **1. Core Vision**

### **Goal**

To build a coordinated, multi-agent "product squad" that can validate and ship early-stage product concepts. The entire process is managed by a "Coordinator" agent with a human (you) in the loop for all key decisions, approvals, and edits.

### **The Stack**

- **Project Path:** `~/git/ai-agent-squad`

- **Agent Framework:** **LangGraph** (Python), to build the squad as a controllable, graph-based state machine.

- **Interface:** A **FastAPI** backend serving a **simple web UI** (HTML/JS/CSS) for project intake and Human-in-the-Loop (HITL) approvals.
- **UI Experience:** The intake page now mirrors the workflow: status pill, timeline chips that light up per agent, real-time response messaging, and per-artifact controls (research/PRD/stories collapsibles plus UX preview buttons that open the Mermaid flow and Tailwind wireframe in full-browser tabs).

- **IDE:** **VS Code** (with some **AI** extension for support).

- **Repository:** A **GitHub** repository for version control.

### **Key Design Principles**

1. **Serialized & Turn-Based:** Only one agent runs at a time. The Coordinator invokes a specialist, which runs, shuts down, and passes its output back.

2. **Human-in-the-Loop (HITL):** The system *must* pause and wait for human approval at key milestones. The web UI is the "control dashboard" for this.

3. **Scoped "Shipped" Definition:** "Shipped" is defined as a complete package of artifacts: a validated product brief, a `README.md`, Mermaid.js user flows, HTML/Tailwind wireframes, and a single-file prototype (e.g., a Python script).


## **2. Architecture: The "Hybrid" Model**

This is the most critical part of our setup. We are balancing performance with isolation.

- **LLM Engine (Host):** **Ollama runs as a native macOS App.**

- **Rationale:** This gives us 100% access to the M2's GPU (Metal) for massive performance. The model `deepseek-r1:8b-0528-qwen3-q4_K_M` powers reasoning agents and lives on the host, while `qwen2.5-coder:7b-instruct-q4_K_M` handles UX/Engineering/QA calls.

- **Agent Application (Docker):** **Our Python code (FastAPI, LangGraph) runs inside a Docker container.**

- **Rationale:** This gives us a clean, isolated, and reproducible environment for our Python libraries. We don't "pollute" our Mac, and the setup is defined in `Dockerfile` and `requirements.txt`.

- **The "Bridge":** The `app` container communicates with the native Ollama app via the special Docker DNS name: `http://host.docker.internal:11434`.

- **Persistence (`tasks.db`):** A **SQLite database** (`tasks.db`) stores the high-level status of each task (e.g., `starting`, `pending_approval`, `completed`). This is used by the web UI to track overall progress.

- **Graph Persistence (`checkpoints.sqlite`):** LangGraph's **`AsyncSqliteSaver`** (with `aiosqlite`) runs as the checkpointer. It records the detailed graph state in `checkpoints.sqlite` so the workflow can pause/resume without losing context, even while the backend remains fully async.

- **Monitoring UI:** A `/tasks_dashboard` page shows every row in `tasks.db`, updating automatically so humans can audit the workflow at any time.

- **Perplexity Research:** When `PERPLEXITY_API_KEY` is set, the Research agent calls the Perplexity `sonar-pro` model for structured findings (summary, opportunities, risks, references), falling back to DuckDuckGo if the API call fails.

- **Product Agent:** Builds on the research to generate a concise PRD and user stories using the local Ollama model, each requiring HITL approval before handing off to UX.

- **Logging:** The application uses Python's standard `logging` module to provide structured, timestamped output. This is crucial for debugging the asynchronous and multi-step workflows.



## **3. The "Product Squad" (Agent Roles)**

1. **The Coordinator (LangGraph Logic):** The "project manager." Routes tasks to specialists.

2. **The Human (You):** The "Decision Maker." Approves/rejects/edits key outputs via the web UI.

3. **Research Agent:** Specialist that calls Perplexity (`sonar-pro`) for structured findings and falls back to DuckDuckGo when needed.

4. **Product Agent:** Specialist that writes product specs and user stories.

5. **UX/Designer Agent:** Specialist that generates **Mermaid.js** user flows and **HTML/Tailwind CSS** wireframes.

6. **Engineering Agent:** Specialist that writes the backend logic and prototype code.

7. **QA Agent:** Specialist that reviews the generated code for bugs and quality.

8. **GTM (Go-To-Market) Agent:** Specialist that writes the final `README.md`.


## **4. Project Plan (Phased Approach)**

**Phase 1: The "Stack" - (COMPLETED)**

- **Goal:** Establish the hybrid architecture.

- **Actions:**

1. Set up Git repo and `.gitignore`.

2. Created `docker-compose.yml` for the standalone `app_service`.

3. Created `Dockerfile` and `requirements.txt`.

4. **Decision:** Swapped `llama3` for `deepseek-r1:8b-0528-qwen3-q4_K_M` for better reasoning.

5. **Decision:** Moved Ollama to a native macOS app for GPU performance.

6. Updated `app.py` to call `http://host.docker.internal:11434/api/generate` directly with schema-backed prompts so every agent gets structured JSON from the local Ollama model.

7. **Decision:** Set up a local `.venv` to solve Pylance warnings.

8. Confirmed a fast, successful "Hello, World!" test.


**Phase 2: The Core Loop - "Coordinator" & HITL UI - (COMPLETED & REFINED)**

- **Goal:** Build a robust, persistent `Intake -> Approve -> End` loop.

- **Actions:**

1.  Built the simplest LangGraph app (`app.py`) with an `intake` and `approved` node.
2.  Created the FastAPI backend with `/start_task`, `/get_pending_approval`, and `/respond_to_approval` endpoints.
3.  Created a basic `index.html` UI for task intake and HITL approval.
4.  **Architectural Refactor:** Replaced the fragile, manual state management with LangGraph's built-in **`AsyncSqliteSaver` checkpointer** to match the FastAPI async runtime.
7.  Added a dedicated `/tasks_dashboard` that renders a live view of `tasks.db`, removing the need to inspect the DB via CLI, and refreshed the intake UI with a card layout, workflow status row, and artifact previews so operators can work from a single screen.
5.  **Separation of Concerns:** The system now uses two databases:
    *   `tasks.db`: For the web application's high-level task status.
    *   `checkpoints.sqlite`: For the LangGraph agent's detailed execution state.
6.  This new architecture dramatically simplifies the code and makes the pause/resume cycle much more reliable.


**Phase 3: The First Specialist - "Research" Agent - (COMPLETED)**

- **Goal:** Add the "Research" agent and give it a tool.

- **Actions:**

1. Added Perplexity integration (`sonar-pro`) with structured JSON output (summary, opportunities, risks, references) plus a DuckDuckGo fallback.

2. Inserted a `research` node into the LangGraph flow ahead of the intake step, storing summaries in both checkpoint state and `tasks.db`.

3. Updated the web UI and `/tasks_dashboard` to display the research findings during HITL approval.

4. Documented environment configuration for `PERPLEXITY_API_KEY` so the integration can run inside Docker.


**Phase 4: The Design Sprint - "Product" & "UX" Agents - (COMPLETED)**

- **Goal:** Add the "Product" and "UX/Designer" agents.

- **Actions:**

1. Product agent drafts a concise PRD (exec summary, market opportunity, needs, scope, success) and pauses for HITL approval for each milestone.

2. After approval, the Product agent generates user stories + acceptance criteria and pauses again for HITL sign-off.

3. UX agent now generates Mermaid user flows and feeds the same context into the Tailwind wireframe generator so both artifacts stay in sync.

4. The dashboard and intake UI were updated with buttons that open flow/wireframe previews in a new tab; `/tasks_dashboard` and CSV exports now include the `user_flow_diagram` and `wireframe_html` columns, keeping UX artifacts visible everywhere.

5. The dashboard and intake UI has a new version where the HITL could collaborate with the agents, actually reviewing/ editing each artifact once its ready for approval. WHen editing the artifact, the UI opens an overlay edit box with options to `Save` or `Cancel`.


**Phase 5: The Build Sprint - "Engineering" & "QA" Agents - (COMPLETED)**

- **Goal:** Add the "Engineering" and "QA" agents to write and review code.

- **Actions:**

1. Added `engineering` and `qa_review` nodes. Engineering generates a single-file FastAPI prototype (filename + code) using the coding model.

2. Implemented the QA code-review loop with `pending_engineering_approval` and `pending_qa_approval` gating before handing off to GTM.

**Phase 6: The "Ship" - "GTM" & Final Output**

- **Goal:** Add the "GTM" agent and package all artifacts.

- **Actions:**

1. Add the `GTM_Agent` to write the `README.md`.

2. Configure the graph to save all artifacts to a `dist/` folder.
