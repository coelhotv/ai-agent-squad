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

*   **⚙️ Graph Persistence (`checkpoints.sqlite`):** LangGraph's built-in **`SqliteSaver`** is used as a **checkpointer**. It automatically saves the detailed internal state of the running agent graph into a separate `checkpoints.sqlite` file. This allows the graph to be paused and resumed reliably.

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

Add the following line to your `requirements.txt` file. This is required for LangGraph's SQLite checkpointer.

```

langgraph[sqlite]

```



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



### Step 5: (Optional) Set Up IDE Support

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



3.  **Intake Node & Pause:** The graph executes the `intake` node. It then hits the predefined `interrupt_before=["approved"]` condition and pauses. The `SqliteSaver` checkpointer automatically saves the complete state of the graph to the `checkpoints.sqlite` file.



4.  **Update for UI:** After the graph pauses, the `/start_task` endpoint updates the task's row in `tasks.db` to `pending_approval` and returns.



5.  **UI Asks for Approval:** The web UI, which polls the backend, finds the pending task and displays the "Approve" / "Reject" buttons.



6.  **Human Decision:** You click **"Approve"**. Your decision is sent to the `/respond_to_approval` endpoint.



7.  **Workflow Resumes:** The backend invokes the graph again, passing the same `task_id` as the `thread_id`. The `SqliteSaver` checkpointer automatically finds the saved state for that thread and resumes the graph exactly where it left off.



8.  **Workflow Completes:** The `approved` node runs, and the graph finishes. The endpoint then updates the task's status in `tasks.db` to `completed`, and the workflow ends for good.



## 5. Project Plan (Phased)



*   **Phase 1: The "Stack" - (COMPLETED)**

    *   Set up hybrid Docker/Ollama architecture.

    *   Established GPU-accelerated LLM connection.



*   **Phase 2: The Core Loop - (COMPLETED & REFINED)**

    *   Built the essential `Intake -> Approve -> End` loop.

    *   Created the FastAPI backend and a basic `index.html` UI.

    *   **Refactored the core workflow to use a persistent `SqliteSaver` checkpointer, dramatically simplifying the logic and improving reliability.**

    *   Separated application state (`tasks.db`) from graph execution state (`checkpoints.sqlite`).



*   **Phase 3: The First Specialist - (NEXT)**

    *   Adding the "Research" agent with a web search tool.



*   **Phase 4: The Design Sprint**

    *   Adding "Product" (specs) and "UX/Designer" (Mermaid, HTML) agents.



*   **Phase 5: The Build Sprint**

    *   Adding "Engineering" (code) and "QA" (review) agents.



*   **Phase 6: The "Ship"**

    *   Adding the "GTM" (README) agent and packaging the final output.
