# Project: Multi-Agent Product Squad

## **1. Core Vision**

### **Goal**

To build a coordinated, multi-agent "product squad" that can validate and ship early-stage product concepts. The entire process is managed by a "Coordinator" agent with a human (you) in the loop for all key decisions, approvals, and edits.

### **The Stack**

- **Project Path:** `~/git/ai-agent-squad`

- **Agent Framework:** **LangGraph** (Python), to build the squad as a controllable, graph-based state machine.

- **Interface:** A **FastAPI** backend serving a **simple web UI** (HTML/JS/CSS) for project intake and Human-in-the-Loop (HITL) approvals.

- **IDE:** **VS Code** (with the **Google Gemini** extension for support).

- **Repository:** A **GitHub** repository for version control.

### **Key Design Principles**

1. **Serialized & Turn-Based:** Only one agent runs at a time. The Coordinator invokes a specialist, which runs, shuts down, and passes its output back.

2. **Human-in-the-Loop (HITL):** The system *must* pause and wait for human approval at key milestones. The web UI is the "control dashboard" for this.

3. **Scoped "Shipped" Definition:** "Shipped" is defined as a complete package of artifacts: a validated product brief, a `README.md`, Mermaid.js user flows, HTML/Tailwind wireframes, and a single-file prototype (e.g., a Python script).


## **2. Architecture: The "Hybrid" Model**

This is the most critical part of our setup. We are balancing performance with isolation.

- **LLM Engine (Host):** **Ollama runs as a native macOS App.**

- **Rationale:** This gives us 100% access to the M2's GPU (Metal) for massive performance. The model `deepseek-r1:8b-llama-distill-q4_K_M` (our chosen model) runs on the host, not in Docker.

- **Agent Application (Docker):** **Our Python code (FastAPI, LangGraph) runs inside a Docker container.**

- **Rationale:** This gives us a clean, isolated, and reproducible environment for our Python libraries. We don't "pollute" our Mac, and the setup is defined in `Dockerfile` and `requirements.txt`.

- **The "Bridge":** The `app` container communicates with the native Ollama app via the special Docker DNS name: `http://host.docker.internal:11434`.

- **Persistence (`tasks.db`):** A **SQLite database** (`tasks.db`) stores the high-level status of each task (e.g., `starting`, `pending_approval`, `completed`). This is used by the web UI to track overall progress.

- **Graph Persistence (`checkpoints.sqlite`):** LangGraph's built-in **`SqliteSaver`** is used as a **checkpointer**. It automatically saves the detailed, internal state of the running agent graph into a separate `checkpoints.sqlite` file. This allows the graph to be reliably paused and resumed without losing its context.

- **Logging:** The application uses Python's standard `logging` module to provide structured, timestamped output. This is crucial for debugging the asynchronous and multi-step workflows.



## **3. The "Product Squad" (Agent Roles)**

1. **The Coordinator (LangGraph Logic):** The "project manager." Routes tasks to specialists.

2. **The Human (You):** The "Decision Maker." Approves/rejects/edits key outputs via the web UI.

3. **Research Agent:** Specialist with a web search tool to validate the idea.

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

4. **Decision:** Swapped `llama3` for `deepseek-r1:8b-llama-distill-q4_K_M` for better reasoning.

5. **Decision:** Moved Ollama to a native macOS app for GPU performance.

6. Updated `app.py` to use `ChatOllama` and connect to `http://host.docker.internal:11434`.

7. **Decision:** Set up a local `.venv` to solve Pylance warnings.

8. Confirmed a fast, successful "Hello, World!" test.

**Phase 2: The Core Loop - "Coordinator" & HITL UI - (COMPLETED & REFINED)**

- **Goal:** Build a robust, persistent `Intake -> Approve -> End` loop.

- **Actions:**

1.  Built the simplest LangGraph app (`app.py`) with an `intake` and `approved` node.
2.  Created the FastAPI backend with `/start_task`, `/get_pending_approval`, and `/respond_to_approval` endpoints.
3.  Created a basic `index.html` UI for task intake and HITL approval.
4.  **Architectural Refactor:** Replaced the fragile, manual state management with LangGraph's built-in **`SqliteSaver` checkpointer**.
5.  **Separation of Concerns:** The system now uses two databases:
    *   `tasks.db`: For the web application's high-level task status.
    *   `checkpoints.sqlite`: For the LangGraph agent's detailed execution state.
6.  This new architecture dramatically simplifies the code and makes the pause/resume cycle much more reliable.

**Phase 3: The First Specialist - "Research" Agent**

- **Goal:** Add the "Research" agent and give it a tool.

- **Actions:**

1. Add `duckduckgo-search` library.

2. Add a "Research" node to the LangGraph.

3. Update the UI to display research findings.

**Phase 4: The Design Sprint - "Product" & "UX" Agents**

- **Goal:** Add the "Product" and "UX/Designer" agents.

- **Actions:**

1. Add `Product_Agent` and `UX_Designer_Agent` nodes.

2. Design prompts for generating user stories, Mermaid.js, and HTML/Tailwind.

**Phase 5: The Build Sprint - "Engineering" & "QA" Agents**

- **Goal:** Add the "Engineering" and "QA" agents to write and review code.

- **Actions:**

1. Add `Engineering_Agent` and `QA_Agent` nodes.

2. Implement the code-review-approval loop.

**Phase 6: The "Ship" - "GTM" & Final Output**

- **Goal:** Add the "GTM" agent and package all artifacts.

- **Actions:**

1. Add the `GTM_Agent` to write the `README.md`.

2. Configure the graph to save all artifacts to a `dist/` folder.
