# Multi-Agent Product Squad

This project is a hands-on experiment to build a coordinated, multi-agent "product squad" capable of validating and prototyping early-stage product ideas. The entire system runs on a local Mac, utilizing a hybrid architecture to balance performance and environment isolation.

This project is being built with the support of Gemini.

## 1. Core Vision

The goal is to create a "Coordinator" agent that manages a team of specialist agents (Research, Product, UX, Engineering, QA, GTM). The system is built with a human-in-the-loop (HITL) for all key decisions, managed via a simple web UI.

* **Coordinator:** The "brain" that routes tasks.
* **Specialists:** Agents with specific skills and tools.
* **Human:** The "approver" who validates and guides the process.

## 2. Architecture: The Hybrid Model

This project uses a hybrid architecture to get the best of both worlds:

* **🤖 LLM Engine (Host):** **Ollama** runs as a **native macOS app**. This provides full access to the M2's Metal GPU for high-performance inference.
* **🐍 Agent Application (Docker):** The core Python application (using **LangGraph** and **FastAPI**) runs in an **isolated Docker container**.
* **🌉 The Bridge:** The `app` container communicates with the native Ollama app via the DNS name `http://host.docker.internal:11434`.

[Image of a diagram showing a Docker container pointing to a host macOS icon]

## 3. Local Development Setup

This project is designed to run on macOS with Docker Desktop and a native Ollama installation.

### Prerequisites
* [Ollama macOS App](https://ollama.com/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Python 3.11+](https://www.python.org/)
* [FastAPI](https://fastapi.tiangolo.com/)

### Step 1: Clone & Set Up the Repo
```bash
# Clone your repository (if it's on GitHub)
# git clone ...
cd ai-agent-squad
```

### Step 2: Set Up the Host Environment
We use a local Python virtual environment to make VS Code's Pylance happy.
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install libraries for IDE support
pip3 install -r requirements.txt

# Tell VS Code to use this venv
# (Cmd+Shift+P) > "Python: Select Interpreter" > ./.venv/bin/python
```

### Step 3: Prepare Ollama (Native)
Make sure the Ollama macOS app is running.
```bash
# Pull the model for the native app
ollama pull deepseek-r1:8b-llama-distill-q4_K_M
```

### Step 4: Launch the Application (Docker)
This command builds and starts the `app_service` Docker container.
```bash
# Build and run the docker container in detached mode
docker-compose up -d --build
```

### Step 5: Test the Connection
You can test that the container can talk to the host's Ollama.
```bash
# Run the 'app.py' script inside the container
docker exec -it app_service python3 app.py
```

This should return a 3-step plan from the DeepSeek model, very quickly.


## 4. Project Plan (Phased)

### Phase 1: The "Stack" (Completed)
- Set up hybrid Docker/Ollama architecture.
- Established GPU-accelerated LLM connection.
- Solved local IDE environment.

### Phase 2: The Core Loop (_NEXT_)
- Building the LangGraph coordinator and FastAPI server.
- Creating the basic index.html UI for intake and approval.

### Phase 3: The First Specialist
- Adding the "Research" agent with a web search tool.

### Phase 4: The Design Sprint
- Adding "Product" (specs) and "UX/Designer" (Mermaid, HTML) agents.

### Phase 5: The Build Sprint
- Adding "Engineering" (code) and "QA" (review) agents.

### Phase 6: The "Ship"
- Adding the "GTM" (README) agent and packaging the final output.
