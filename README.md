# Project: Multi-Agent Product Squad

This project is a hands-on experiment to build a coordinated, multi-agent "product squad" capable of validating and prototyping early-stage product ideas. The system is built to be resilient, using a local database for persistence and structured logging for debuggability.

The squad is managed by a **LangGraph**-based "Coordinator" and includes a **FastAPI** backend to serve a simple web UI for Human-in-the-Loop (HITL) approvals.

## 1. Core Vision

The goal is to create a "Coordinator" agent that manages a team of specialist agents (Research, Product, UX, Engineering, QA, GTM). The system is built with a human-in-the-loop (HITL) for all key decisions, managed via a simple web UI.

*   **Coordinator:** The "brain" that routes tasks.
*   **Specialists:** Agents with specific skills and tools.
*   **Human:** The "approver" who validates and guides the process.

## 2. Architecture: The Hybrid & Persistent Model



This project uses a hybrid architecture to get the best of both worlds: performance and isolation. The entire system is designed for persistence, ensuring that no work is lost.



*   **🤖 LLM Engine (Host):** **Ollama** runs as a **native macOS app**. This provides full access to the M2's Metal GPU for high-performance inference.

*   **🐍 Agent Application (Docker):** The core Python application (using **LangGraph** and **FastAPI**) runs in an **isolated Docker container**.

*   **🌉 The Bridge:** The `app` container communicates with the native Ollama app via the DNS name `http://host.docker.internal:11434`.

*   **💾 Application Persistence (`tasks.db`):** A **SQLite database** stores the high-level state of each task (e.g., `starting`, `pending_approval`, `completed`). This database is used by the web UI to track progress.

*   **⚙️ Graph Persistence (`checkpoints.sqlite`):** LangGraph's **`AsyncSqliteSaver`** (with `aiosqlite`) is used as the checkpointer. It keeps the graph's internal state in a `checkpoints.sqlite` file so the workflow can pause and resume asynchronously without losing context.

*   **🧭 Research Layer:** The Research agent first queries Perplexity's `sonar-pro` model for a structured JSON report and automatically falls back to DuckDuckGo (via `ddgs`) if the API is unavailable.

*   **📝 Product Agent:** A lightweight PM agent uses the local Ollama model (default `deepseek-r1:8b-llama-distill-q4_K_M`) to draft a PRD and user stories with dual HITL checkpoints before the UX agent picks up the baton. Perplexity is only used as a fallback.

*   **📜 Logging:** The application uses Python's standard `logging` module to provide structured, informative output for easier debugging.



## 3. How to Run the Application



### Prerequisites

*   [Ollama macOS App](https://ollama.com/) (and ensure it's running)

*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) (and ensure it's running)

*   Python 3.11+



### Step 1: Prepare the Model

Make sure the Ollama macOS app is running and you have pulled the model we're using.

```bash

ollama pull deepseek-r1:8b-llama-distill-q4_K_M

```



### Step 2: Update Dependencies







Ensure the `requirements.txt` file includes the packages for SQLite checkpointing. These are required for LangGraph's async checkpointer.







```



langgraph-checkpoint-sqlite
aiosqlite
ddgs
httpx



```







**Note:** We initially used `langgraph[sqlite]`, but this was found to cause dependency resolution issues. The correct combination is `langgraph-checkpoint-sqlite` plus `aiosqlite`. The Research agent also depends on the `ddgs` package for fallback searches and `httpx` for the Perplexity API client.

### Step 2.1: Configure Perplexity (Optional but Recommended)

If you have a Perplexity API key, export it so the Research agent can call the `sonar-pro` model. Without it, the system falls back to DuckDuckGo.

```bash
export PERPLEXITY_API_KEY="pplx-..."
```

When running via Docker Compose, set the same variable in your shell or `.env` so it propagates to the container.



### Step 2.2: Configure Ollama (Optional)

By default the Product agent uses the local `deepseek-r1:8b-llama-distill-q4_K_M` model via the native Ollama app. Set `OLLAMA_MODEL` (and optionally `OLLAMA_BASE_URL`) if you want to change the model or host:

```bash
export OLLAMA_MODEL="llama3.1:8b"
export OLLAMA_BASE_URL="http://host.docker.internal:11434"
```

If you skip this step, the defaults defined in `app.py` are used.



### Step 3: Clear Old Databases (Important)

To prevent errors from previous versions, delete these files from your project folder if they exist:

*   `tasks.db`

*   `checkpoints.sqlite`



### Step 4: Launch the Application

This single command builds the Docker image, installs the Python dependencies, and starts the FastAPI server.

```bash

docker-compose up -d --build

```

The server will be running and accessible at `http://localhost:8000`. If you exported `PERPLEXITY_API_KEY`, Docker Compose will pass it into the container automatically (see `docker-compose.yml`).

### Step 5: Monitor Tasks

Visit `http://localhost:8000/tasks_dashboard` to view every row currently stored in `tasks.db`. The dashboard auto-refreshes so you can keep an eye on task status without using the terminal.



### Step 6: (Optional) Set Up IDE Support

For better IDE support (e.g., in VS Code), you can create a local virtual environment and install the dependencies there. This allows Pylance to find the libraries.

```bash

# Create and activate the virtual environment

python3 -m venv .venv

source .venv/bin/activate



# Install libraries

pip3 install -r requirements.txt

```



## 4. How It Works: The Refactored Core Loop



The application now uses a much simpler and more robust workflow that relies on a LangGraph checkpointer.



1.  **Submit an Idea:** You navigate to `http://localhost:8000`, enter a product idea, and click **"Start Task"**.



2.  **Kick-off the Process:** The browser sends the idea to the `/start_task` endpoint. The backend:

    a. Creates a unique `task_id`.

    b. Saves a new task to the `tasks.db` database with a `starting` status.

    c. Invokes the LangGraph workflow with the `task_id` as the `thread_id`.



3.  **Research Phase:** The Research agent calls Perplexity (`sonar-pro`) for a structured JSON report (summary, opportunities, risks, references). If the API is unavailable, it falls back to DuckDuckGo and summarizes the top findings, then stores the result in the graph state.



4.  **Product Agent – PRD Draft:** Using the research output, the Product agent (running on the local Ollama model) writes a concise PRD (exec summary, opportunity, needs, scope, success criteria) and the system pauses for human approval.



5.  **UI Update:** The `/start_task` endpoint persists the PRD (and research summary) to `tasks.db` and returns. The web UI surfaces both artifacts and asks for approval.



6.  **Human Decision (PRD):** Once approved, the graph resumes and the Product agent generates user stories + acceptance criteria. If you reject at any point, the task is marked as such and stored in the DB for later review.



7.  **Product Agent – User Stories:** After approval, the Product agent writes 3–4 stories with acceptance criteria plus a mini backlog. The workflow pauses again for human review while the UI shows the updated artifacts in collapsible sections.



8.  **Second Approval:** Approving this step resumes the graph, which marks the task `ready_for_ux` so the next specialist can pick it up. Every pause/resume is handled by `AsyncSqliteSaver`, so the graph state is always persisted.



## 5. Project Plan (Phased)



*   **Phase 1: The "Stack" - (COMPLETED)**

    *   Set up hybrid Docker/Ollama architecture.

    *   Established GPU-accelerated LLM connection.



*   **Phase 2: The Core Loop - (COMPLETED & REFINED)**

    *   Built the essential `Intake -> Approve -> End` loop.

    *   Created the FastAPI backend and a basic `index.html` UI.

    *   **Refactored the core workflow to use a persistent `AsyncSqliteSaver` checkpointer, dramatically simplifying the logic and improving reliability.**

    *   Separated application state (`tasks.db`) from graph execution state (`checkpoints.sqlite`).



*   **Phase 3: The First Specialist - (COMPLETED)**

    *   Added the Research agent that calls Perplexity (`sonar-pro`) for structured summaries (summary, opportunities, risks, references) and automatically falls back to DuckDuckGo when needed.
    *   Persist research results in both LangGraph state and `tasks.db`, then surface them in the HITL UI and `/tasks_dashboard` for approval context.



*   **Phase 4: The Design Sprint - (IN PROGRESS)**

    *   Added the Product agent that creates the PRD and initial user stories with two HITL checkpoints before handing off to UX.
    *   Next up: add the UX/Designer agent (Mermaid user flow + HTML/Tailwind wireframes).



*   **Phase 5: The Build Sprint**

    *   Adding "Engineering" (code) and "QA" (review) agents.



*   **Phase 6: The "Ship"**

    *   Adding the "GTM" (README) agent and packaging the final output.
