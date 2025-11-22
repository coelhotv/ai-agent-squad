# AI Product Squad Architecture

This document outlines the high-level architecture of the AI Product Squad application, focusing on the agentic workflow managed by LangGraph.

## Core Components

- **FastAPI Backend (`app.py`):** Serves the API, manages task state in a SQLite database, and orchestrates the agent workflow.
- **LangGraph:** The core engine that defines and executes the sequence of agent tasks as a state machine.
- **Ollama:** Provides the LLM inference for the agents (e.g., `deepseek-coder`, `qwen`).
- **SQLite:** Used for two purposes:
  1. `tasks.db`: A simple application database to store the status and artifacts of each task for the UI.
  2. `checkpoints.sqlite`: The persistence layer for LangGraph, allowing workflows to be paused and resumed.
- **Vanilla JS Frontend (`index.html`, `tasks.html`):** A simple user interface for submitting ideas, providing human-in-the-loop (HITL) approvals, and viewing artifacts.

## Agent Workflow

The workflow is a linear sequence of nodes, with a human approval step (interruption) between each node.

1.  **Research Agent:** Conducts initial market research.
2.  **Product Agent (PRD):** Creates a Product Requirements Document.
3.  **Product Agent (Stories):** Generates User Stories and Acceptance Criteria.
4.  **UX Agent:** Designs a user flow diagram (MermaidJS) and a low-fidelity wireframe (HTML/Tailwind).
5.  **Architect Agent (Spec):** This is a two-step process:
    - **Reasoning (`deepseek`):** First, a reasoning model creates a high-level plan (`architect_reasoning`) for the API structure based on the user stories.
    - **Contract Generation (`qwen-coder`):** Second, a coding model takes the reasoning plan as input and generates the detailed JSON specification (`engineering_spec`) for the API schemas and endpoints.
6.  **Developer Agent:** Implements the API specification in a runnable Python (FastAPI) script.
7.  **QA Agents:** Each generation step (spec and code) is followed by an internal QA review to ensure quality and adherence to requirements before presenting for human approval.