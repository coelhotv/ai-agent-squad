# Setup & Prerequisites

## Prerequisites

Ensure your machine meets the following before launching the stack:

- **Ollama macOS app:** Install from [https://ollama.com](https://ollama.com) and keep it running so the native GPU-backed models are available.
- **Docker Desktop:** Required for the FastAPI/LangGraph container defined in `docker-compose.yml`.
- **Python 3.11+:** Needed if you run parts of the project locally outside Docker.

## Pull the Models

Ollama hosts the LLMs locally. Run:

```bash
ollama pull deepseek-r1:8b-0528-qwen3-q4_K_M
ollama pull qwen2.5-coder:7b-instruct-q6_K
```

The reasoning tasks (PM + GTM agents) prefer `deepseek-r1:8b-0528-qwen3-q4_K_M`; UX, Engineering, and QA prompts rely on `qwen2.5-coder:7b-instruct-q6_K`.

## Install Dependencies (Optional, for local dev)

```bash
pip install -r requirements.txt
```

The stack explicitly depends on:

- `langgraph-checkpoint-sqlite`
- `aiosqlite`
- `ddgs`
- `httpx`

These packages support LangGraph checkpoints, DuckDuckGo fallbacks, and Perplexity calls.

## Environment Variables

- `PERPLEXITY_API_KEY` (optional): When set, the Research agent calls Perplexity’s `sonar-pro` model for structured JSON summaries. Without it, DuckDuckGo is used.
- `OLLAMA_REASONING_MODEL` (optional): Defaults to `deepseek-r1:8b-0528-qwen3-q4_K_M`.
- `OLLAMA_CODING_MODEL` (optional): Defaults to `qwen2.5-coder:7b-instruct-q6_K`.
- `OLLAMA_MODEL` (optional): Legacy fallback for whichever model you want to return if the role-specific values are unset.
- `OLLAMA_BASE_URL` (optional): Defaults to `http://host.docker.internal:11434`. Change only if your host exposes Ollama differently.

Copy `.env.example` to `.env` and update values there. Docker Compose automatically loads `.env`, so you can tweak configs without rebuilding. For ad-hoc overrides, you can still export them in your shell.

## Optional: Resetting State

`tasks.db` and `checkpoints.sqlite` reside inside the Docker volume named `app_data` (mounted at `/data`), so you should not delete them from your host workspace. To wipe state:

```bash
# stop services and remove the persistent volume (data lives in app_data)
docker-compose down -v
```

Run this only when you intentionally need to discard progress (schema changes, stuck approvals, etc.). The stack otherwise maintains progress across restarts via `checkpoints.sqlite`.

## Launch the Stack

```bash
docker-compose up -d --build
```

This builds the image, installs dependencies, and starts the FastAPI server. Visit `http://localhost:8000` to see the intake UI and `http://localhost:8000/tasks_dashboard` for the live dashboard.

If you exported `PERPLEXITY_API_KEY`, Docker Compose will pass it through automatically (see the `environment` block in `docker-compose.yml`).

## Optional Local Development

For IDE support (VS Code, Pylance, etc.):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This lets you open the repo in VS Code while still using Docker for runtime execution.

When you edit `app.py` locally, restart the Docker service (or run FastAPI directly with `uvicorn app:app --reload` outside Docker) so changes take effect.
