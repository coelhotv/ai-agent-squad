# Project: Multi-Agent Product Squad

This project is a hands-on experiment to build a coordinated, multi-agent "product squad" capable of validating and prototyping early-stage product ideas. The system is built to be resilient, using a local database for persistence and structured logging for debuggability.

The squad is managed by a **LangGraph**-based "Coordinator" and includes a **FastAPI** backend to serve a simple web UI for Human-in-the-Loop (HITL) approvals.

## 1. Core Vision

The goal is to create a "Coordinator" agent that manages a team of specialist agents (Research, Product, UX, Engineering, QA, GTM). The system is built with a human-in-the-loop (HITL) for all key decisions, managed via a simple web UI.

*   **Coordinator:** The "brain" that routes tasks.
*   **Specialists:** Agents with specific skills and tools.
*   **Human:** The "approver" who validates and guides the process.

## 2. Architecture: The Hybrid Model

This project uses a hybrid architecture to get the best of both worlds:

*   **🤖 LLM Engine (Host):** **Ollama** runs as a **native macOS app**. This provides full access to the M2's Metal GPU for high-performance inference.
*   **🐍 Agent Application (Docker):** The core Python application (using **LangGraph** and **FastAPI**) runs in an **isolated Docker container**.
*   **🌉 The Bridge:** The `app` container communicates with the native Ollama app via the DNS name `http://host.docker.internal:11434`.
*   **💾 Persistence:** A **SQLite database** (`tasks.db`) runs inside the container to store all task states, ensuring no data is lost if the application restarts.
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

### Step 2: Launch the Application
This single command builds the Docker image, installs the Python dependencies, and starts the FastAPI server.
```bash
docker-compose up -d --build
```
The server will be running and accessible at `http://localhost:8000`.

### Step 3: (Optional) Set Up IDE Support
For better IDE support (e.g., in VS Code), you can create a local virtual environment and install the dependencies there. This allows Pylance to find the libraries.
```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install libraries
pip3 install -r requirements.txt

# In VS Code, use Cmd+Shift+P -> "Python: Select Interpreter"
# and choose the one in the ./.venv/bin directory.
```

## 4. How It Works: The Core Loop

The application is now running the essential `Intake -> Approve -> End` loop. Here’s a walkthrough:

1.  **Submit an Idea:** You navigate to `http://localhost:8000`, enter a product idea into the text box, and click **"Start Task"**.

2.  **Kick-off the Process:** The browser sends the idea to the backend. The backend creates a unique Task ID, **saves the new task to the database** with a `starting` status, and triggers the LangGraph workflow.

3.  **Intake Node & Pause:** The first agent node runs, updates the task state to `pending_approval`, and **updates the database**. The workflow then **pauses** to wait for a human decision.

4.  **UI Asks for Approval:** The web UI, which has been checking with the backend, finds the pending task. It displays the approval message and shows the **"Approve"** and **"Reject"** buttons.

5.  **Human Decision:** You click **"Approve"**. Your decision is sent back to the backend.

6.  **Workflow Completes:** The backend receives your approval and tells the LangGraph workflow to continue. It runs the final nodes, which mark the task as `completed`. The **final status is saved to the database**, and the workflow ends.

## 5. Project Plan (Phased)

*   **Phase 1: The "Stack" - (COMPLETED)**
    *   Set up hybrid Docker/Ollama architecture.
    *   Established GPU-accelerated LLM connection.
    *   Solved local IDE environment.

*   **Phase 2: The Core Loop - (COMPLETED)**
    *   Built the essential `Intake -> Approve -> End` loop.
    *   Built the simplest LangGraph app (`app.py`).
    *   Added a **SQLite database** for persistent task storage.
    *   Implemented **structured logging** for improved debugging.
    *   Created the FastAPI backend and a basic `index.html` UI for the HITL flow.

*   **Phase 3: The First Specialist - (NEXT)**
    *   Adding the "Research" agent with a web search tool.

*   **Phase 4: The Design Sprint**
    *   Adding "Product" (specs) and "UX/Designer" (Mermaid, HTML) agents.

*   **Phase 5: The Build Sprint**
    *   Adding "Engineering" (code) and "QA" (review) agents.

*   **Phase 6: The "Ship"**
    *   Adding the "GTM" (README) agent and packaging the final output.