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



```







**Note:** We initially used `langgraph[sqlite]`, but this was found to cause dependency resolution issues. The correct combination is `langgraph-checkpoint-sqlite` plus `aiosqlite`. The Research agent also depends on the `duckduckgo-search` package, which is already listed in `requirements.txt`.



### Step 3: Clear Old Databases (Important)

To prevent errors from previous versions, delete these files from your project folder if they exist:

*   `tasks.db`

*   `checkpoints.sqlite`



### Step 4: Launch the Application

This single command builds the Docker image, installs the Python dependencies, and starts the FastAPI server.

```bash

docker-compose up -d --build

```

The server will be running and accessible at `http://localhost:8000`.

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



3.  **Research Phase:** The Research agent queries DuckDuckGo for the product idea, summarizes the top findings, and stores that summary in the graph state.



4.  **Intake Node & Pause:** The graph executes the `intake` node. It then hits the predefined `interrupt_before=["approved"]` condition and pauses. The async checkpointer automatically saves the complete state of the graph to the `checkpoints.sqlite` file.



5.  **Update for UI:** After the graph pauses, the `/start_task` endpoint updates the task's row in `tasks.db` to `pending_approval`, persists the research summary, and returns.



6.  **UI Asks for Approval:** The web UI, which polls the backend, finds the pending task, shows the research findings, and displays the "Approve" / "Reject" buttons.



7.  **Human Decision:** You click **"Approve"**. Your decision is sent to the `/respond_to_approval` endpoint.



8.  **Workflow Resumes:** The backend invokes the graph again, passing the same `task_id` as the `thread_id`. `AsyncSqliteSaver` automatically finds the saved state for that thread and resumes the graph exactly where it left off.



9.  **Workflow Completes:** The `approved` node runs, and the graph finishes. The endpoint then updates the task's status in `tasks.db` to `completed`, and the workflow ends for good.



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

    *   Added the Research agent that calls DuckDuckGo, summarizes findings, and stores them in both LangGraph state and `tasks.db`.
    *   Updated the HITL UI to display research summaries before approval and surfaced the same data on the `/tasks_dashboard`.



*   **Phase 4: The Design Sprint**

    *   Adding "Product" (specs) and "UX/Designer" (Mermaid, HTML) agents.



*   **Phase 5: The Build Sprint**

    *   Adding "Engineering" (code) and "QA" (review) agents.



*   **Phase 6: The "Ship"**

    *   Adding the "GTM" (README) agent and packaging the final output.
