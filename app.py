import os
import json
import csv
from io import StringIO
# --- Core Imports ---
import uuid
import logging
from contextlib import AsyncExitStack

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Pydantic Imports ---
from pydantic import BaseModel

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- Typing Imports ---
from typing import TypedDict, Optional, List

# --- Database Imports ---
from sqlalchemy import create_engine, Column, String, Text, text
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# --- Research Imports ---
from ddgs import DDGS
import httpx
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Reduce SQL logging noise (especially from dashboard polling)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

from app_settings import get_settings

settings = get_settings()

# --- Application Database (SQLite) ---
# This DB stores the high-level status of tasks for the UI
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(String, primary_key=True, index=True)
    product_idea = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    pending_approval_content = Column(Text, nullable=True)
    research_summary = Column(Text, nullable=True)
    prd_summary = Column(Text, nullable=True)
    user_stories = Column(Text, nullable=True)
    user_flow_diagram = Column(Text, nullable=True)
    wireframe_html = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

def ensure_column(column_name: str):
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("tasks")]
    if column_name not in columns:
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE tasks ADD COLUMN {column_name} TEXT"))

for col_name in ("research_summary", "prd_summary", "user_stories", "user_flow_diagram", "wireframe_html"):
    ensure_column(col_name)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- LangGraph State and Checkpointer ---
class AgentState(TypedDict):
    task_id: str
    product_idea: str
    status: str
    pending_approval_content: Optional[str]
    research_summary: Optional[str]
    prd_summary: Optional[str]
    user_stories: Optional[str]
    user_flow_diagram: Optional[str]
    wireframe_html: Optional[str]

# Keep the async connection open for the lifetime of the app and close on shutdown.
_checkpointer_stack = AsyncExitStack()
PERPLEXITY_API_KEY = settings.perplexity_api_key
PERPLEXITY_API_URL = str(settings.perplexity_api_url)
OLLAMA_BASE_URL = str(settings.ollama_base_url)
OLLAMA_MODEL = settings.ollama_model
memory_saver: Optional[AsyncSqliteSaver] = None
app_graph = None


def log_environment_status():
    """Log useful environment and configuration details so operators know what's enabled."""
    logger.info("Database URL: %s", settings.database_url)
    checkpoints_path = settings.checkpoints_path
    logger.info("Checkpoint store: %s", checkpoints_path)
    logger.info("Ollama base URL: %s", OLLAMA_BASE_URL)
    logger.info("Ollama model: %s", OLLAMA_MODEL)

    if PERPLEXITY_API_KEY:
        masked_key = PERPLEXITY_API_KEY[:6] + "..." if len(PERPLEXITY_API_KEY) > 6 else "***"
        logger.info("Perplexity API key detected (%s). Research agent will attempt sonar-pro.", masked_key)
    else:
        logger.warning(
            "PERPLEXITY_API_KEY is not set. Research agent will rely on DuckDuckGo fallback only."
        )

    data_dir = os.path.dirname(settings.checkpoints_path) or "/data"
    if not os.path.exists(data_dir):
        logger.warning("Data directory %s does not exist. Attempting to create it.", data_dir)
        try:
            os.makedirs(data_dir, exist_ok=True)
            logger.info("Created missing data directory at %s.", data_dir)
        except OSError as exc:
            logger.error("Failed to create data directory %s: %s", data_dir, exc)

    if not os.access(data_dir, os.W_OK):
        logger.warning("Data directory %s is not writable. SQLite persistence may fail.", data_dir)

    try:
        with httpx.Client(timeout=2) as client:
            response = client.get(f"{OLLAMA_BASE_URL}api/ps")
            response.raise_for_status()
            logger.info("Successfully reached Ollama endpoint (%s).", f"{OLLAMA_BASE_URL}api/ps")
    except Exception as exc:
        logger.warning("Unable to reach Ollama at %s: %s", OLLAMA_BASE_URL, exc)


def run_research_query(product_idea: str) -> str:
    """Use Perplexity for research; fall back to DuckDuckGo if unavailable."""
    if PERPLEXITY_API_KEY:
        try:
            structured = query_perplexity(product_idea)
            if structured:
                return structured
        except Exception as exc:
            logger.exception("Perplexity research failed for '%s'", product_idea)

    return run_duckduckgo_research(product_idea)


def query_perplexity(product_idea: str) -> Optional[str]:
    json_schema = {
        "name": "market_research",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "opportunities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "details": {"type": "string"},
                        },
                        "required": ["title", "details"],
                    },
                },
                "risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "details": {"type": "string"},
                        },
                        "required": ["title", "details"],
                    },
                },
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["source", "url"],
                    },
                },
            },
            "required": ["summary", "opportunities", "risks", "references"],
        },
    }
    system_prompt = (
        "You are a senior market research analyst. Analyze the idea, compare competitors,"
        " and highlight opportunities, risks, and references."
    )
    user_prompt = (
        f"Idea: {product_idea}. Provide insights for PM/UX planning."
        " Keep it concise and structured. Max 2-3 paragraphs for summary,"
        " 2-3 bullets each for opportunities and risks."
        " Use credible sources and cite URLs in references."
    )

    parsed = call_perplexity_json(system_prompt, user_prompt, json_schema)
    if not parsed:
        return None

    return format_structured_summary(parsed)


def call_perplexity_json(
    system_prompt: str, user_prompt: str, json_schema: dict
) -> Optional[dict]:
    if not PERPLEXITY_API_KEY:
        return None

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": json_schema},
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    with httpx.Client(timeout=120) as client:
        response = client.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            logger.error(
                "Perplexity API error %s: %s",
                response.status_code,
                response.text[:500],
            )
        response.raise_for_status()
        data = response.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        return None

    return _extract_json(content)


def call_ollama_json(
    system_prompt: str, user_prompt: str, schema_description: str
) -> Optional[dict]:
    if not OLLAMA_MODEL:
        return None

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{user_prompt}\nReturn ONLY valid JSON matching: {schema_description}",
            },
        ],
        "stream": False,
        "think": True,
    }

    try:
        with httpx.Client(timeout=180) as client:
            response = client.post(f"{OLLAMA_BASE_URL}api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.exception("Ollama request failed: %s", exc)
        return None

    content = data.get("message", {}).get("content")
    if not content:
        return None

    return _extract_json(content)


def _extract_json(content: str) -> Optional[dict]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def format_structured_summary(data: dict) -> str:
    lines = []
    summary = data.get("summary")
    if summary:
        lines.append("Summary:\n" + summary.strip())

    opportunities = data.get("opportunities") or []
    if opportunities:
        lines.append("\nOpportunities:")
        for opp in opportunities:
            title = opp.get("title") or "Opportunity"
            details = opp.get("details") or ""
            lines.append(f"- {title}: {details}")

    risks = data.get("risks") or []
    if risks:
        lines.append("\nRisks:")
        for risk in risks:
            title = risk.get("title") or "Risk"
            details = risk.get("details") or ""
            lines.append(f"- {title}: {details}")

    refs = data.get("references") or []
    if refs:
        lines.append("\nReferences:")
        for ref in refs:
            source = ref.get("source") or "Source"
            url = ref.get("url") or ""
            lines.append(f"- {source}: {url}")

    return "\n".join(lines).strip()


def generate_prd_document(product_idea: str, research_summary: str | None) -> str:
    schema_description = (
        "{\"executive_summary\": string, \"market_opportunity\": [string],"
        " \"customer_needs\": [string], \"product_scope\": [string],"
        " \"success_criteria\": [string]}"
    )
    system_prompt = (
        "You are a senior PM drafting a one-page PRD. Keep it punchy and actionable."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nResearch summary: {research_summary or 'n/a'}."
        " Draft the requested sections with the most important bullets first."
    )
    parsed = call_ollama_json(system_prompt, user_prompt, schema_description)
    if not parsed and PERPLEXITY_API_KEY:
        json_schema = {
            "name": "prd_outline",
            "schema": {
                "type": "object",
                "properties": {
                    "executive_summary": {"type": "string"},
                    "market_opportunity": {"type": "array", "items": {"type": "string"}},
                    "customer_needs": {"type": "array", "items": {"type": "string"}},
                    "product_scope": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "executive_summary",
                    "market_opportunity",
                    "customer_needs",
                    "product_scope",
                    "success_criteria",
                ],
            },
        }
        parsed = call_perplexity_json(system_prompt, user_prompt, json_schema)
    if not parsed:
        return fallback_prd(product_idea, research_summary)

    lines = ["Executive Summary:", parsed.get("executive_summary", "")]
    lines.append("\nMarket Opportunity & Positioning:")
    for item in parsed.get("market_opportunity", [])[:3]:
        lines.append(f"- {item}")
    lines.append("\nCustomer Needs:")
    for item in parsed.get("customer_needs", [])[:4]:
        lines.append(f"- {item}")
    lines.append("\nProduct Scope & Use Cases:")
    for item in parsed.get("product_scope", [])[:4]:
        lines.append(f"- {item}")
    lines.append("\nSuccess Criteria:")
    for item in parsed.get("success_criteria", [])[:3]:
        lines.append(f"- {item}")
    return "\n".join(lines).strip()


def fallback_prd(product_idea: str, research_summary: str | None) -> str:
    return (
        f"Executive Summary:\nA first iteration of '{product_idea}' focused on a single"
        " core workflow.\n\nMarket Opportunity & Positioning:\n"
        f"- Based on research: {research_summary or 'insights pending.'}\n"
        "- Target early adopters and gather feedback.\n\nCustomer Needs:\n"
        "- Simple onboarding\n- Clear value communication\n- Feedback loop\n\n"
        "Product Scope & Use Cases:\n- Pilot use case that validates demand\n"
        "- Internal admin dashboard for basic tracking\n\nSuccess Criteria:\n"
        "- 10 pilot sign-ups\n- 60% repeat usage in 2 weeks\n- Qualitative feedback on usability"
    )


def generate_user_stories(product_idea: str, prd_summary: str | None) -> str:
    schema_description = (
        "{\"stories\":[{\"title\":string,\"story\":string,"
        "\"acceptance_criteria\":[string]}],\"backlog\":[string]}"
    )
    system_prompt = (
        "You are a senior PM writing crisp user stories with acceptance criteria."
        " Keep scope tight for v0 and stay within the provided PRD."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nPRD context: {prd_summary or 'Unavailable.'}\n"
        " Produce 3-4 user stories plus a short backlog list."
    )
    parsed = call_ollama_json(system_prompt, user_prompt, schema_description)
    if not parsed and PERPLEXITY_API_KEY:
        json_schema = {
            "name": "product_user_stories",
            "schema": {
                "type": "object",
                "properties": {
                    "stories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "story": {"type": "string"},
                                "acceptance_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["title", "story", "acceptance_criteria"],
                        },
                    },
                    "backlog": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["stories", "backlog"],
            },
        }
        parsed = call_perplexity_json(system_prompt, user_prompt, json_schema)
    if not parsed:
        return fallback_user_stories(product_idea)

    lines = ["User Stories:"]
    for entry in parsed.get("stories", [])[:4]:
        title = entry.get("title") or "Story"
        story_text = entry.get("story") or ""
        lines.append(f"\n{title}\n{story_text}")
        for criteria in entry.get("acceptance_criteria", [])[:3]:
            lines.append(f"  - AC: {criteria}")

    backlog = parsed.get("backlog", [])
    if backlog:
        lines.append("\nNext Iteration Backlog:")
        for item in backlog[:3]:
            lines.append(f"- {item}")

    return "\n".join(lines).strip()


def fallback_user_stories(product_idea: str) -> str:
    return (
        "User Stories:\n\n1. As a pilot customer I want to try the core flow so that I can"
        f" see how '{product_idea}' helps me.\n  - AC: Account created\n"
        "  - AC: Key task completed\n\n2. As an operator I want a simple"
        " dashboard so that I can monitor usage.\n  - AC: View list of active"
        " users\n  - AC: Export activity\n\nNext Iteration Backlog:\n- Social"
        " proof content\n- Referral workflow"
    )


def generate_user_flow_diagram(product_idea: str, user_stories: str | None) -> str:
    schema_description = "{\"mermaid\":string,\"notes\":string}"
    system_prompt = (
        "You are a senior UX designer, producing a concise Mermaid user flow with nodes and decision points."
        " Focus on the main user stories and acceptance criteria to outline the key journey."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nUser stories:\n{user_stories or 'Unavailable.'}\n"
        " Return Mermaid flowchart text with 4-6 nodes max. A single code block only."
    )
    parsed = call_ollama_json(system_prompt, user_prompt, schema_description)
    if parsed and parsed.get("mermaid"):
        return parsed["mermaid"]
    return fallback_user_flow_diagram(product_idea)


def fallback_user_flow_diagram(product_idea: str) -> str:
    return (
        "flowchart TD\n"
        "  Awareness[User hears about product] --> Landing[Visit idea page]\n"
        "  Landing --> Evaluate{Fits need?}\n"
        "  Evaluate -- yes --> Submit[Submit pilot request]\n"
        "  Evaluate -- no --> Collect[Collect feedback]\n"
        "  Submit --> Squad[Squad reviews request]\n"
        "  Squad --> Onboard[Onboard + kickoff]\n"
        f"  Onboard --> Value[Experience '{product_idea}' value]\n"
    )


def generate_wireframe_html(product_idea: str, user_stories: str | None) -> str:
    schema_description = "{\"layout\":string}"
    system_prompt = (
        "You are a technical product designer delivering low-fidelity HTML + Tailwind CSS markup."
        " Use semantic sections, headings, and placeholder CTAs to produce a clean wireframe layout for the following idea."
        " Focus on key journeys and actions from the user stories."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nUser stories:\n{user_stories or 'Unavailable.'}\n"
        " Output only HTML markup, max ~60 lines, ready for a Tailwind sandbox, no comments."
    )
    parsed = call_ollama_json(system_prompt, user_prompt, schema_description)
    if parsed and parsed.get("layout"):
        return parsed["layout"]
    return fallback_wireframe_html(product_idea)


def fallback_wireframe_html(product_idea: str) -> str:
    return (
        "<div class=\"min-h-screen bg-slate-900 text-slate-100 p-8\">\n"
        "  <header class=\"max-w-4xl mx-auto bg-slate-800 px-6 py-4 rounded-2xl\">\n"
        f"    <p class=\"text-sm uppercase tracking-widest text-slate-400\">Concept</p>\n"
        f"    <h1 class=\"text-2xl font-semibold\">{product_idea}</h1>\n"
        "  </header>\n"
        "  <main class=\"max-w-4xl mx-auto mt-6 grid gap-4 md:grid-cols-2\">\n"
        "    <section class=\"bg-slate-800 rounded-2xl p-4\">\n"
        "      <h2 class=\"text-lg font-medium mb-2\">Key Journey</h2>\n"
        "      <ul class=\"space-y-2 text-sm\">\n"
        "        <li class=\"flex items-start gap-2\"><span class=\"mt-1 h-2 w-2 rounded-full bg-emerald-400\"></span>Discover opportunity</li>\n"
        "        <li class=\"flex items-start gap-2\"><span class=\"mt-1 h-2 w-2 rounded-full bg-emerald-400\"></span>Submit pilot request</li>\n"
        "        <li class=\"flex items-start gap-2\"><span class=\"mt-1 h-2 w-2 rounded-full bg-emerald-400\"></span>Track deliverables</li>\n"
        "      </ul>\n"
        "    </section>\n"
        "    <section class=\"bg-slate-800 rounded-2xl p-4\">\n"
        "      <h2 class=\"text-lg font-medium mb-2\">Actions</h2>\n"
        "      <div class=\"space-y-3\">\n"
        "        <button class=\"w-full rounded-xl bg-emerald-500/20 px-4 py-3 text-left\">Start Task</button>\n"
        "        <button class=\"w-full rounded-xl bg-slate-700 px-4 py-3 text-left\">Review Tasks</button>\n"
        "      </div>\n"
        "    </section>\n"
        "  </main>\n"
        "</div>\n"
    )


def run_duckduckgo_research(product_idea: str) -> str:
    """Fallback DuckDuckGo search with multiple strategies."""
    def format_results(results):
        summary_lines = []
        for idx, result in enumerate(results, start=1):
            title = (result.get("title") or "Untitled Result").strip()
            snippet = (result.get("body") or result.get("snippet") or "").strip()
            url = result.get("href") or result.get("url") or result.get("link") or ""
            line = f"{idx}. {title}"
            if snippet:
                line += f" — {snippet}"
            if url:
                line += f" ({url})"
            summary_lines.append(line)
        return "\n".join(summary_lines)

    def search(query, source="text"):
        with DDGS() as ddgs:
            if source == "text":
                return list(
                    ddgs.text(
                        query,
                        max_results=5,
                        region="us-en",
                        safesearch="off",
                        timelimit="y",
                        backend="html",
                    )
                )
            elif source == "news":
                return list(
                    ddgs.news(
                        query,
                        max_results=5,
                        region="us-en",
                        safesearch="off",
                    )
                )
            return []

    queries = [
        f"{product_idea} competitors",
        f"{product_idea} market news",
        f"market research for {product_idea}",
    ]

    try:
        for idx, query in enumerate(queries):
            source = "text" if idx != 1 else "news"
            results = search(query, source=source)
            if results:
                return format_results(results)
    except Exception as exc:
        logger.exception("DuckDuckGo fallback failed for idea '%s'", product_idea)
        return f"Research unavailable due to error: {exc}"

    return "No public findings were returned for this idea."

# --- Agent Node Functions ---
def research_node(state: AgentState):
    logger.info(f"--- Node: research_node (Task ID: {state['task_id']}) ---")
    state['research_summary'] = run_research_query(state['product_idea'])
    state['status'] = "pending_research_approval"
    state['pending_approval_content'] = (
        f"Research completed for '{state['product_idea']}'. Review the findings and approve to start the PRD draft."
    )
    return state


def product_prd_node(state: AgentState):
    logger.info(f"--- Node: product_prd_node (Task ID: {state['task_id']}) ---")
    state['prd_summary'] = generate_prd_document(
        state['product_idea'], state.get('research_summary')
    )
    state['status'] = "pending_prd_approval"
    state['pending_approval_content'] = (
        "Review the draft PRD, make edits if needed, and approve to generate user stories."
    )
    return state


def product_stories_node(state: AgentState):
    logger.info(f"--- Node: product_stories_node (Task ID: {state['task_id']}) ---")
    state['user_stories'] = generate_user_stories(
        state['product_idea'], state.get('prd_summary')
    )
    state['status'] = "pending_story_approval"
    state['pending_approval_content'] = (
        "Review the initial user stories & AC. Approve to hand off to UX."
    )
    return state


def ux_design_node(state: AgentState):
    logger.info(f"--- Node: ux_design_node (Task ID: {state['task_id']}) ---")
    state['user_flow_diagram'] = generate_user_flow_diagram(
        state['product_idea'], state.get('user_stories')
    )
    state['wireframe_html'] = generate_wireframe_html(
        state['product_idea'], state.get('user_stories')
    )
    state['status'] = "pending_ux_approval"
    state['pending_approval_content'] = (
        "Review the UX flow and wireframe. Approve to hand off to engineering."
    )
    return state


def approved_node(state: AgentState):
    logger.info(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "ready_for_engineering"
    state['pending_approval_content'] = None
    return state

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("product_prd", product_prd_node)
workflow.add_node("product_stories", product_stories_node)
workflow.add_node("ux_design", ux_design_node)
workflow.add_node("approved", approved_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "product_prd")
workflow.add_edge("product_prd", "product_stories")
workflow.add_edge("product_stories", "ux_design")
workflow.add_edge("ux_design", "approved")
workflow.add_edge("approved", END)


async def initialize_graph():
    """Initialize the LangGraph checkpointer and compiled workflow."""
    global memory_saver, app_graph
    memory_saver = await _checkpointer_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(settings.checkpoints_path)
    )
    app_graph = workflow.compile(
        checkpointer=memory_saver,
        interrupt_before=["product_prd", "product_stories", "ux_design", "approved"]
    )


def get_app_graph():
    if app_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not initialized.")
    return app_graph

# --- FastAPI Application ---
app = FastAPI(title="Multi-Agent Product Squad API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    log_environment_status()
    await initialize_graph()
    logger.info("LangGraph initialized with AsyncSqliteSaver.")


@app.on_event("shutdown")
async def on_shutdown():
    await _checkpointer_stack.aclose()
    logger.info("LangGraph resources closed.")

# --- Pydantic Models for API ---
class StartTaskRequest(BaseModel):
    product_idea: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    product_idea: str
    pending_approval_content: Optional[str] = None
    research_summary: Optional[str] = None
    prd_summary: Optional[str] = None
    user_stories: Optional[str] = None
    user_flow_diagram: Optional[str] = None
    wireframe_html: Optional[str] = None
    class Config:
        from_attributes = True

class RespondToApprovalRequest(BaseModel):
    task_id: str
    approved: bool

# --- API Endpoints (Refactored) ---
@app.post("/start_task", response_model=TaskStatus)
async def start_task(request: StartTaskRequest, db: Session = Depends(get_db)):
    logger.info(f"Received request to start task for idea: '{request.product_idea[:50]}...'")
    graph = get_app_graph()
    task_id = str(uuid.uuid4())
    
    # 1. Create the task in our application DB
    db_task = Task(
        task_id=task_id,
        product_idea=request.product_idea,
        status="starting",
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    logger.info(f"Task {task_id} created in DB with status 'starting'.")

    # 2. Define the graph's starting state and config
    initial_state = AgentState(
        task_id=task_id,
        product_idea=request.product_idea,
        status="starting",
        pending_approval_content=None,
        research_summary=None,
        prd_summary=None,
        user_stories=None,
        user_flow_diagram=None,
        wireframe_html=None,
    )
    config = {"configurable": {"thread_id": task_id}}

    # 3. Asynchronously invoke the graph. It will run until it hits the interruption.
    # The checkpointer automatically saves its state.
    await graph.ainvoke(initial_state, config)
    logger.info(f"Graph for task {task_id} executed until interruption.")

    # 4. Get the state of the graph at the interruption point
    interrupted_state = await graph.aget_state(config)
    logger.info(f"Current graph state for task {task_id}: {interrupted_state.values}")

    # 5. Update our application DB with the new status from the graph
    db_task.status = interrupted_state.values['status']
    db_task.pending_approval_content = interrupted_state.values['pending_approval_content']
    db_task.research_summary = interrupted_state.values.get('research_summary')
    db_task.prd_summary = interrupted_state.values.get('prd_summary')
    db_task.user_stories = interrupted_state.values.get('user_stories')
    db_task.user_flow_diagram = interrupted_state.values.get('user_flow_diagram')
    db_task.wireframe_html = interrupted_state.values.get('wireframe_html')
    db.commit()
    db.refresh(db_task)
    logger.info(f"Task {task_id} updated in DB to status '{db_task.status}'.")

    return TaskStatus.from_orm(db_task)

@app.get("/get_pending_approval", response_model=Optional[TaskStatus])
def get_pending_approval(db: Session = Depends(get_db)):
    pending_statuses = [
        "pending_research_approval",
        "pending_prd_approval",
        "pending_story_approval",
        "pending_ux_approval",
        "pending_approval",  # backward compatibility
    ]
    pending_task = (
        db.query(Task)
        .filter(Task.status.in_(pending_statuses))
        .order_by(Task.task_id)
        .first()
    )
    if pending_task:
        return TaskStatus.from_orm(pending_task)
    return None

@app.get("/tasks", response_model=List[TaskStatus])
def list_tasks(db: Session = Depends(get_db)):
    tasks = db.query(Task).order_by(Task.task_id.desc()).all()
    return [TaskStatus.from_orm(task) for task in tasks]


@app.get("/tasks/export")
def export_tasks(db: Session = Depends(get_db)):
    tasks = db.query(Task).order_by(Task.task_id.desc()).all()
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "Task ID",
            "Product Idea",
            "Status",
            "Research Summary",
            "PRD Summary",
            "User Stories",
            "User Flow Diagram",
            "Wireframe HTML",
            "Pending Approval Content",
        ]
    )
    for task in tasks:
        writer.writerow(
            [
                task.task_id,
                task.product_idea,
                task.status,
                task.research_summary or "",
                task.prd_summary or "",
                task.user_stories or "",
                task.user_flow_diagram or "",
                task.wireframe_html or "",
                task.pending_approval_content or "",
            ]
        )
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tasks_export.csv"},
    )

@app.post("/respond_to_approval", response_model=TaskStatus)
async def respond_to_approval(request: RespondToApprovalRequest, db: Session = Depends(get_db)):
    pending_statuses = {
        "pending_research_approval",
        "pending_prd_approval",
        "pending_story_approval",
        "pending_ux_approval",
        "pending_approval",
    }
    db_task = db.query(Task).filter(Task.task_id == request.task_id).first()
    if not db_task or db_task.status not in pending_statuses:
        raise HTTPException(status_code=404, detail="Task not found or not pending approval.")

    if not request.approved:
        db_task.status = 'rejected'
        db_task.pending_approval_content = None
        db.commit()
        logger.info(f"Task {request.task_id} rejected by human.")
        return TaskStatus.from_orm(db_task)

    logger.info(f"Task {request.task_id} approved by human. Resuming graph.")
    
    # 1. Define the config to resume the correct graph instance
    config = {"configurable": {"thread_id": request.task_id}}
    graph = get_app_graph()
    
    # 2. Invoke the graph again. The checkpointer loads the state automatically.
    # We pass `None` as the state because the checkpointer is handling it.
    final_state = await graph.ainvoke(None, config)
    logger.info(f"Graph for task {request.task_id} advanced to state '{final_state.get('status')}'.")

    # 3. Update our application DB with the new status/content
    db_task.status = final_state.get('status', db_task.status)
    db_task.pending_approval_content = final_state.get('pending_approval_content')
    db_task.research_summary = final_state.get('research_summary', db_task.research_summary)
    db_task.prd_summary = final_state.get('prd_summary', db_task.prd_summary)
    db_task.user_stories = final_state.get('user_stories', db_task.user_stories)
    db_task.user_flow_diagram = final_state.get('user_flow_diagram', db_task.user_flow_diagram)
    db_task.wireframe_html = final_state.get('wireframe_html', db_task.wireframe_html)
    db.commit()
    logger.info(f"Task {request.task_id} updated in DB to final status '{db_task.status}'.")
    
    return TaskStatus.from_orm(db_task)

# --- Static File and Status Endpoints ---
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse("index.html")

@app.get("/status")
def read_root():
    return {"message": "Welcome to the Multi-Agent Product Squad API!"}

@app.get("/tasks_dashboard", include_in_schema=False)
async def read_tasks_dashboard():
    return FileResponse("tasks.html")
