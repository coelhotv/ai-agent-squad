import os
import json
import csv
from io import StringIO
# --- Core Imports ---
import uuid
import logging
import time
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
    engineering_file_name = Column(Text, nullable=True)
    engineering_code = Column(Text, nullable=True)
    qa_review = Column(Text, nullable=True)
    engineering_spec = Column(Text, nullable=True)
    engineering_qa = Column(Text, nullable=True)
    last_rejected_step = Column(Text, nullable=True)
    last_rejected_at = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def ensure_column(column_name: str):
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("tasks")]
    if column_name not in columns:
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE tasks ADD COLUMN {column_name} TEXT"))

for col_name in (
    "research_summary",
    "prd_summary",
    "user_stories",
    "user_flow_diagram",
    "wireframe_html",
    "engineering_file_name",
    "engineering_code",
    "qa_review",
    "engineering_spec",
    "engineering_qa",
    "last_rejected_step",
    "last_rejected_at",
):
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
    engineering_file_name: Optional[str]
    engineering_code: Optional[str]
    qa_review: Optional[str]
    engineering_spec: Optional[str]
    engineering_qa: Optional[str]
    last_rejected_step: Optional[str]

# Keep the async connection open for the lifetime of the app and close on shutdown.
_checkpointer_stack = AsyncExitStack()
PERPLEXITY_API_KEY = settings.perplexity_api_key
PERPLEXITY_API_URL = str(settings.perplexity_api_url)
OLLAMA_BASE_URL = str(settings.ollama_base_url)
OLLAMA_REASONING_MODEL = settings.ollama_reasoning_model
OLLAMA_CODING_MODEL = settings.ollama_coding_model
OLLAMA_MODEL = settings.ollama_model
memory_saver: Optional[AsyncSqliteSaver] = None
app_graph = None


def log_environment_status():
    """Log useful environment and configuration details so operators know what's enabled."""
    logger.info("Database URL: %s", settings.database_url)
    checkpoints_path = settings.checkpoints_path
    logger.info("Checkpoint store: %s", checkpoints_path)
    logger.info("Ollama base URL: %s", OLLAMA_BASE_URL)
    logger.info("Ollama reasoning model: %s", OLLAMA_REASONING_MODEL or OLLAMA_MODEL)
    logger.info("Ollama coding model: %s", OLLAMA_CODING_MODEL or OLLAMA_MODEL)

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
                "target_audience": {"type": "string", "description": "A brief description of the ideal customer profile."},
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
            "required": ["summary", "target_audience", "opportunities", "risks"],
        },
    }
    system_prompt = (
        "You are a senior market research analyst. Analyze the idea, compare competitors,"
        " and highlight opportunities, risks, and references. Output strictly using the provided JSON schema."
        " Keep it concise and structured."
    )
    user_prompt = (
        f"Idea: {product_idea}. Provide insights for PM/UX planning."
        " Max 2-3 paragraphs for summary, 2-3 bullets each for opportunities and risks."
        " Use credible sources and list main URLs used as references (3-5 max)."
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

    start_time = time.time()
    with httpx.Client(timeout=90) as client:
        response = client.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            logger.error(
                "Perplexity API error %s: %s",
                response.status_code,
                response.text[:500],
            )
        response.raise_for_status()
        data = response.json()

    duration = time.time() - start_time
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    logger.info(
        f"Perplexity call took {duration:.2f}s. "
        f"Tokens: Input={input_tokens}, Output={output_tokens}"
    )

    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        return None

    return _extract_json(content)


def _get_ollama_model(reasoning: bool) -> Optional[str]:
    model = OLLAMA_REASONING_MODEL if reasoning else OLLAMA_CODING_MODEL
    if not model:
        model = OLLAMA_MODEL
    return model


def call_ollama_json(
    system_prompt: str,
    user_prompt: str,
    json_schema: object,
    *,
    reasoning: bool = False,
) -> Optional[dict]:
    model = _get_ollama_model(reasoning)
    if not model:
        logger.warning("No Ollama model configured for reasoning=%s.", reasoning)
        return None

    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": json_schema,
        "stream": False,
        "think": reasoning,
    }

    logger.info("Calling Ollama model %s (reasoning=%s).", model, reasoning)

    try:
        start_time = time.time()
        with httpx.Client(timeout=300) as client:
            response = client.post(f"{OLLAMA_BASE_URL}api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        duration = time.time() - start_time

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        logger.info(
            f"Ollama call took {duration:.2f}s. "
            f"Tokens: Input={input_tokens}, Output={output_tokens}"
        )
    except Exception as exc:
        logger.exception("Ollama request failed: %s", exc)
        return None

    content = data.get("response")
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

    target_audience = data.get("target_audience")
    if target_audience:
        lines.append("\nTarget Audience:\n" + target_audience.strip())

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
    prd_schema = {
        "type": "object",
        "properties": {
            "app_name": {"type": "string"},
            "executive_summary": {"type": "string"},
            "market_opportunity": {
                "type": "array",
                "items": {"type": "string"}
            },
            "customer_needs": {
                "type": "array",
                "items": {"type": "string"}
            },
            "product_scope": {
                "type": "array",
                "items": {"type": "string"}
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": [
            "app_name",
            "executive_summary",
            "market_opportunity",
            "customer_needs",
            "product_scope",
            "success_criteria",
        ]
    }    
    system_prompt = (
        "You are a senior PM translating user needs and business opportunities into a clear, "
        "concise, and actionable one-page PRD for engineering and design teams. Keep it punchy."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nResearch summary: {research_summary or 'n/a'}."
        " Draft the requested sections with the most important bullets first."
    )
    parsed = call_ollama_json(
        system_prompt, user_prompt, prd_schema, reasoning=True
    )
    if not parsed and PERPLEXITY_API_KEY:
        json_schema = {
            "name": "prd_outline",
            "schema": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string"},
                    "executive_summary": {"type": "string"},
                    "market_opportunity": {"type": "array", "items": {"type": "string"}},
                    "customer_needs": {"type": "array", "items": {"type": "string"}},
                    "product_scope": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "app_name",
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

    lines = ["App:", parsed.get("app_name", "")]
    lines.append("\nExecutive Summary:")
    lines.append(parsed.get("executive_summary", ""))
    lines.append("\nMarket Opportunity & Positioning:")
    for item in parsed.get("market_opportunity", [])[:3]:
        lines.append(f"- {item}")
    lines.append("\nCustomer Needs:")
    for item in parsed.get("customer_needs", [])[:4]:
        lines.append(f"- {item}")
    lines.append("\nProduct Scope and Use cases:")
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
        "Product Scope:\n- Pilot use case that validates demand\n"
        "- Internal admin dashboard for basic tracking\n\nSuccess Criteria:\n"
        "- 10 pilot sign-ups\n- 60% repeat usage in 2 weeks\n- Qualitative feedback on usability"
    )


def generate_user_stories(product_idea: str, prd_summary: str | None) -> str:
    stories_schema = {
        "type": "object",
        "properties": {
            "stories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "story": {"type": "string"},
                        "acceptance_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["story", "acceptance_criteria"],
                },
            },
            "backlog": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["stories", "backlog"],
    }
    system_prompt = (
        "You are a senior PM writing crisp user stories with acceptance criteria."
        ' Follow the format "As a [user type], I want [goal] so that [benefit]."'
        " Keep scope tight for MVP (v0) and stay within the provided PRD."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nPRD context: {prd_summary or 'Unavailable.'}\n"
        " Produce 2 user stories to be in the MVP, plus 1-2 user stories for a second release."
        " Number each story sequentially. For each story, provide 2-3 acceptance criteria."
        " End with a short backlog list."
    )
    parsed = call_ollama_json(
        system_prompt, user_prompt, stories_schema, reasoning=True
    )
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
                                "story": {"type": "string"},
                                "acceptance_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["story", "acceptance_criteria"],
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
        story_text = entry.get("story") or ""
        lines.append(f"\n{story_text}")
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
    mermaid_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "nodes_list": {
                "type": "array",
                "minItems": 4,
                "maxItems": 6,
                "items": {
                    "type": "string",
                    "description": "Unique ID or label for the node"
                },
                "description": "List exactly 4 to 6 distinct steps/nodes for the diagram."
            },
            "mermaid_syntax": {
                "type": "string",
                "description": "The Mermaid code. It must ONLY use the nodes defined in 'nodes_list'."
            }
        },
        "required": ["title", "nodes_list", "mermaid_syntax"]
    }    
    system_prompt = (
        "You are an expert UX Architect. Your goal is to create a clear, logical MermaidJS user flow diagram "
        "based on the provided product idea, its estabished **user stories and acceptance criterias**. "
        "For this MVP release, you can skip the login/registration flow and focus on the 2 first stories/ac."
        "You MUST output a JSON object that adheres strictly to the provided schema."
    )
    user_prompt = f"""
        Idea: {product_idea}.
        User stories:
        {user_stories or 'Unavailable.'}
        ### GUIDELINES:
         1. **Node Count:**
         - Plan exactly 4 to 6 steps. No more, no less.
         - Use the 'nodes_list' field in the JSON to plan these steps first.
         2. **Node Types:**
         - Use standard rectangular nodes `[ ]` for User Actions (e.g., `A[User clicks Login]`).
         - If needed, use diamond nodes `{{ }}` for System Decisions or logic checks (e.g., `B{{ Is Valid? }}`).
         - Ensure the flow has a clear Start and End that commits to the core product idea.
         3. **MermaidJS Syntax Rules:**
         - Use simple, alphanumeric IDs for nodes (e.g., `Step1`, `Step2`). Do NOT use spaces in IDs.
         - Put the descriptive text inside the brackets/parentheses.
         - Example: `Start[User Opens App] --> Check{{ Logged In? }}`.
         - Orientation: Defaults to `graph TD` (Top-Down).
         4. **Content Quality:**
         - Keep labels concise (2-5 words).
    """
    parsed = call_ollama_json(system_prompt, user_prompt, mermaid_schema)
    if parsed and parsed.get("mermaid_syntax"):
        return parsed["mermaid_syntax"]
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


def generate_wireframe_html(product_idea: str, user_stories: str | None, flow_diagram: str | None) -> str:
    wireframe_schema = {
        "type": "object",
        "properties": {
            "html_content": {
                "type": "string",
                "description": "The raw HTML elements. Do NOT include <html>, <head>, or <body> tags. Just the <div> structures."
            }
        },
        "required": ["html_content"]
    }    
    system_prompt = (
        "You are a Senior UI/UX Prototyper. Your goal is to create a low-fidelity wireframe writing HTML and Tailwind CSS code. "
        "Think in terms of reusable components like headers, cards, and sections."
        "Use the user stories, acceptance criteria, and user flow as inputs. "
    )
    user_prompt = f"""
        Idea: {product_idea}.
        
        User stories:
        {user_stories or 'Unavailable.'}

        User flow (Mermaid):
        {flow_diagram or 'Unavailable.'}

        ### DESIGN RULES:
        ## **Aesthetic:** Stick to a low-fidelity wireframe design. 
        - Use 'grayscale' style when possible, but **be mindful of 'dark' mode** on selected systems setups.
        - Guarantee high contrast for readability (e.g., `bg-slate-900`, `text-slate-100`).
        - Use borders (`border`, `border-gray-300`) to define areas.

        ##. **Images:** Do NOT use <img> tags requiring external URLs.
        - Instead, use placeholder divs: `<div class="w-full h-48 bg-gray-300 flex items-center justify-center">Image Placeholder</div>`

        ## **Typography:** Use standard sans-serif. Use `font-bold` for headers.

        4. **Output:** - Provide ONLY the HTML structure (divs, sections, columns) aligned with the user flow and user stories.
        - Do not write the <!DOCTYPE> or <body> tags; just the content inside.
    """
    parsed = call_ollama_json(system_prompt, user_prompt, wireframe_schema)
    if parsed and parsed.get("html_content"):
        return parsed["html_content"]
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


def generate_engineering_spec(
    product_idea: str,
    user_stories: str | None,
    user_flow_diagram: str | None,
) -> str:
    spec_schema = {
        "type": "object",
        "properties": {
            "overview": {
            "type": "string",
            "description": "A brief high-level summary of the architecture choice."
            },
            "database_models": {
            "type": "array",
            "description": "List of Pydantic models representing the data.",
            "items": {
                "type": "object",
                "properties": {
                "name": { "type": "string", "description": "Class name, e.g., 'User'" },
                "fields": {
                    "type": "array",
                    "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "type": { "type": "string", "description": "Python type, e.g., 'str', 'int', 'datetime'" },
                        "constraints": { "type": "string", "description": "e.g., 'gt=0', 'optional', 'unique'" }
                    },
                    "required": ["name", "type"]
                    }
                }
                },
                "required": ["name", "fields"]
            }
            },
            "api_endpoints": {
            "type": "array",
            "description": "List of API routes to implement.",
            "items": {
                "type": "object",
                "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "description": "HTTP Method"
                },
                "path": { "type": "string", "description": "URL path, e.g., '/users/{id}'" },
                "summary": { "type": "string", "description": "What this endpoint does." },
                "request_body_model": {
                    "type": ["string", "null"],
                    "description": "The Pydantic model name used for the input body. NULL if GET/DELETE."
                },
                "response_model": {
                    "type": "string",
                    "description": "The Pydantic model name returned."
                }
                },
                "required": ["method", "path", "summary", "response_model"]
            }
            }
        },
        "required": ["database_models", "api_endpoints"]
    }
    system_prompt = """"
        You are a Senior Software Architect specializing in FastAPI and Pydantic v2.
        Your goal is to design a robust backend API structure based on User Stories/AC and Mermaid flow.

        CRITICAL DESIGN RULES:
        1. **Coverage:** You must design schema and endpoints to cover the initial two (1 & 2) User Stories/AC + flow provided.
        2. **HTTP Semantics:** For any creation or update (POST/PUT), you MUST specify a `request_body_model`. Do NOT design APIs that use Query parameters for inserting data.
        3. **Data Integrity:** Define strict constraints for your models (e.g., positive numbers for counts, required fields).
        4. **Separation:** Create distinct models for Requests (Input) and Responses (Output) if necessary to hide internal IDs or computed fields.
        5. **Conflict Handling:** Ensure paths include all parameters and plan conflict/missing-resource handling.
        
        **Do NOT write implementation logic — only the interface.**
    """
    user_prompt = f"""
    Product idea: {product_idea}
    
    User stories and AC:
    {user_stories or 'Unavailable.'}

    User flow (Mermaid) for context:
    {user_flow_diagram or 'Unavailable.'}

    Now, design the database models and API endpoints needed.
    """
    parsed = call_ollama_json(system_prompt, user_prompt, spec_schema)
    if parsed and parsed.get("database_models") and parsed.get("api_endpoints"):
        return json.dumps(parsed, indent=2)
    return "Spec unavailable"


def generate_engineering_code_from_spec(
    product_idea: str,
    user_stories: str | None,
    wireframe_html: str | None,
    spec_text: str | None,
) -> tuple[str, str]:
    code_schema = {
        "type": "object",
        "properties": {
            "file_name": { "type": "string" },
            "code": { "type": "string", "description": "Complete, runnable Python code." }
        },
        "required": ["file_name", "code"]
    }
    system_prompt = """
        You are an Expert Python Developer.
        Your task is to generate a single-file FastAPI application that implements the provided Architecture Plan EXACTLY.
        User stories and wireframe are provided for context only.

        IMPLEMENTATION RULES:
        1. **Strict Adherence:** Implement every model and endpoint defined in the Architecture Plan. Do not skip any.
        2. **Pydantic V2:** Use `model_validator` or `field_validator` for constraints. Use `BaseModel`.
        3. **FastAPI Best Practices:**
        - Use `CORSMiddleware` allowing ["*"] (restrain origins e.g., ['http://localhost:8000']).
        - Use `logging` to track request activity in handlers, health/status endpoint, and note in-memory storage is for demo only..
        - Ensure `if __name__ == "__main__":` block is present.
        4. Use JSON bodies for POST/PUT/PATCH. Align path params in decorators and signatures.
        5. **No Hallucinations:** Do not add features not requested in the plan.
        6. Return only JSON as specified in the schema, with file_name + code only (no Markdown).

    """
    user_prompt = f"""
    Product idea: {product_idea}
    
    User stories (for context only, spec is authoritative):
    {user_stories or 'Unavailable.'}

    Wireframe (for context only, spec is authoritative):
    {wireframe_html or 'Unavailable.'}

    Architecture Spec (implement exactly these models and endpoints):
    {spec_text or 'Unavailable.'}

    Now write the Python code!
    """
    parsed = call_ollama_json(system_prompt, user_prompt, code_schema)
    if parsed and parsed.get("file_name") and parsed.get("code"):
        return parsed["file_name"], parsed["code"]
    return "main.py", "Code unavailable."


def run_engineering_qa_review(
    product_idea: str,
    user_stories: str | None,
    spec_text: str | None,
    engineering_code: str | None,
) -> str:
    qa_schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "title": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["severity", "title", "details"],
                },
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "findings"],
    }
    system_prompt = (
        "You are a Senior QA Engineer reviewing a FastAPI prototype against the provided spec and user stories. "
        "Output JSON only. Focus on: schema/endpoint alignment with the spec, validation (field_validator, positive numbers, enums), "
        "conflict/missing-resource handling, logging usage, CORS configuration, and noting demo-only storage risks. "
        "Coverage of at least the first user story is required. "
        "Be concise and actionable."
    )
    user_prompt = f"""
    Product idea: {product_idea}
    User stories: {user_stories or 'Unavailable.'}
    Architetural Spec: {spec_text or 'Unavailable.'}

    Prototype code to review:
    {engineering_code or 'Unavailable.'}
    """
    parsed = call_ollama_json(system_prompt, user_prompt, qa_schema)
    if not parsed:
        return "QA review unavailable. No structured response returned."
    lines = [f"Verdict: {parsed.get('verdict', 'unknown')}"]
    findings = parsed.get("findings") or []
    if findings:
        lines.append("Findings:")
        for finding in findings:
            sev = finding.get("severity") or "info"
            title = finding.get("title") or "Issue"
            details = finding.get("details") or ""
            lines.append(f"- [{sev}] {title}: {details}")
    recs = parsed.get("recommendations") or []
    if recs:
        lines.append("Recommendations:")
        for rec in recs:
            lines.append(f"- {rec}")
    return "\n".join(lines).strip()

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
                        backend="duckduckgo",
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
        state['product_idea'], state.get('user_stories'), state.get('user_flow_diagram')
    )
    state['status'] = "pending_ux_approval"
    state['pending_approval_content'] = (
        "Review the UX flow and wireframe. Approve to hand off to engineering."
    )
    return state


def engineering_node(state: AgentState):
    logger.info(f"--- Node: engineering_node (Task ID: {state['task_id']}) ---")
    state['engineering_spec'] = generate_engineering_spec(
        state['product_idea'],
        state.get('user_stories'),
        state.get('user_flow_diagram'),
    )
    file_name, code = generate_engineering_code_from_spec(
        state['product_idea'],
        state.get('user_stories'),
        state.get('wireframe_html'),
        state.get('engineering_spec'),
    )
    state['engineering_file_name'] = file_name
    state['engineering_code'] = code
    state['engineering_qa'] = run_engineering_qa_review(
        state['product_idea'],
        state.get('user_stories'),
        state.get('engineering_spec'),
        state.get('engineering_code'),
    )
    state['status'] = "pending_engineering_bundle_approval"
    state['pending_approval_content'] = (
        "Review the engineering bundle (spec, code, QA). Approve to hand off to GTM."
    )
    return state


def approved_node(state: AgentState):
    logger.info(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "ready_for_gtm"
    state['pending_approval_content'] = None
    return state


# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("product_prd", product_prd_node)
workflow.add_node("product_stories", product_stories_node)
workflow.add_node("ux_design", ux_design_node)
workflow.add_node("engineering", engineering_node)
workflow.add_node("approved", approved_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "product_prd")
workflow.add_edge("product_prd", "product_stories")
workflow.add_edge("product_stories", "ux_design")
workflow.add_edge("ux_design", "engineering")
workflow.add_edge("engineering", "approved")
workflow.add_edge("approved", END)


async def initialize_graph():
    """Initialize the LangGraph checkpointer and compiled workflow."""
    global memory_saver, app_graph
    memory_saver = await _checkpointer_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(settings.checkpoints_path)
    )
    app_graph = workflow.compile(
        checkpointer=memory_saver,
        interrupt_before=[
            "product_prd",
            "product_stories",
            "ux_design",
            "engineering",
            "approved",
        ],
    )


def get_app_graph():
    if app_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not initialized.")
    return app_graph

# --- FastAPI Application ---
async def lifespan(app: FastAPI):
    log_environment_status()
    await initialize_graph()
    logger.info("LangGraph initialized with AsyncSqliteSaver.")
    try:
        yield
    finally:
        await _checkpointer_stack.aclose()
        logger.info("LangGraph resources closed.")


app = FastAPI(title="Multi-Agent Product Squad API", version="1.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    engineering_spec: Optional[str] = None
    engineering_file_name: Optional[str] = None
    engineering_code: Optional[str] = None
    engineering_qa: Optional[str] = None
    last_rejected_step: Optional[str] = None
    last_rejected_at: Optional[str] = None
    class Config:
        from_attributes = True

class ArtifactUpdate(BaseModel):
    task_id: str
    research_summary: Optional[str] = None
    prd_summary: Optional[str] = None
    user_stories: Optional[str] = None
    user_flow_diagram: Optional[str] = None
    wireframe_html: Optional[str] = None
    engineering_spec: Optional[str] = None
    engineering_file_name: Optional[str] = None
    engineering_code: Optional[str] = None
    engineering_qa: Optional[str] = None


class RespondToApprovalRequest(BaseModel):
    task_id: str
    approved: bool
    overrides: ArtifactUpdate | None = None


class ResubmitRequest(BaseModel):
    task_id: str
    step: str  # e.g., research, product_prd, product_stories, ux_design, engineering, qa_review


PENDING_STATUSES = {
    "pending_research_approval",
    "pending_prd_approval",
    "pending_story_approval",
    "pending_ux_approval",
    "pending_engineering_bundle_approval",
    "pending_approval",
}


def apply_artifact_overrides(task_id: str, overrides: ArtifactUpdate, db: Session):
    db_task = db.query(Task).filter(Task.task_id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if db_task.status not in PENDING_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Artifacts can only be edited while the task is pending approval.",
        )
    updated = False
    for field in (
        "research_summary",
        "prd_summary",
        "user_stories",
        "user_flow_diagram",
        "wireframe_html",
        "engineering_spec",
        "engineering_file_name",
        "engineering_code",
        "engineering_qa",
    ):
        value = getattr(overrides, field)
        if value is not None:
            setattr(db_task, field, value)
            updated = True
    if updated:
        db.commit()
        db.refresh(db_task)
    return db_task


STEP_STATUS_MAP = {
    "research": "pending_research_approval",
    "product_prd": "pending_prd_approval",
    "product_stories": "pending_story_approval",
    "ux_design": "pending_ux_approval",
    "engineering": "pending_engineering_bundle_approval",
}


def clear_artifacts_for_step(db_task: Task, step: str):
    """Drop artifacts at and after the specified step so regeneration is clean."""
    downstream_fields = {
        "research": [
            "research_summary",
            "prd_summary",
            "user_stories",
            "user_flow_diagram",
            "wireframe_html",
            "engineering_spec",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "product_prd": [
            "prd_summary",
            "user_stories",
            "user_flow_diagram",
            "wireframe_html",
            "engineering_spec",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "product_stories": [
            "user_stories",
            "user_flow_diagram",
            "wireframe_html",
            "engineering_spec",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "ux_design": [
            "user_flow_diagram",
            "wireframe_html",
            "engineering_file_name",
            "engineering_spec",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "engineering": [
            "engineering_spec",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
    }
    fields = downstream_fields.get(step, [])
    for field in fields:
        setattr(db_task, field, None)


def status_to_step(status: str | None) -> Optional[str]:
    if not status:
        return None
    inverse = {
        "pending_research_approval": "research",
        "pending_prd_approval": "product_prd",
        "pending_story_approval": "product_stories",
        "pending_ux_approval": "ux_design",
        "pending_engineering_bundle_approval": "engineering",
    }
    return inverse.get(status)


def build_state_from_task(db_task: Task) -> AgentState:
    return {
        "task_id": db_task.task_id,
        "product_idea": db_task.product_idea,
        "status": db_task.status,
        "pending_approval_content": db_task.pending_approval_content,
        "research_summary": db_task.research_summary,
        "prd_summary": db_task.prd_summary,
        "user_stories": db_task.user_stories,
        "user_flow_diagram": db_task.user_flow_diagram,
        "wireframe_html": db_task.wireframe_html,
        "engineering_spec": db_task.engineering_spec,
        "engineering_file_name": db_task.engineering_file_name,
        "engineering_code": db_task.engineering_code,
        "engineering_qa": db_task.engineering_qa,
        "last_rejected_step": db_task.last_rejected_step,
    }

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
        engineering_spec=None,
        engineering_file_name=None,
        engineering_code=None,
        engineering_qa=None,
        last_rejected_step=None,
        last_rejected_at=None,
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
    db_task.engineering_spec = interrupted_state.values.get('engineering_spec')
    db_task.engineering_file_name = interrupted_state.values.get('engineering_file_name')
    db_task.engineering_code = interrupted_state.values.get('engineering_code')
    db_task.engineering_qa = interrupted_state.values.get('engineering_qa')
    db_task.last_rejected_step = interrupted_state.values.get('last_rejected_step')
    db_task.last_rejected_at = interrupted_state.values.get('last_rejected_at')
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
        "pending_engineering_bundle_approval",
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


@app.post("/update_artifacts", response_model=TaskStatus)
def update_artifacts(request: ArtifactUpdate, db: Session = Depends(get_db)):
    db_task = apply_artifact_overrides(request.task_id, request, db)
    return TaskStatus.from_orm(db_task)


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
            "Engineering Spec",
            "Engineering File Name",
            "Engineering Code",
            "Engineering QA",
            "Last Rejected Step",
            "Last Rejected At",
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
                task.engineering_spec or "",
                task.engineering_file_name or "",
                task.engineering_code or "",
                task.engineering_qa or "",
                task.last_rejected_step or "",
                task.last_rejected_at or "",
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
        "pending_engineering_bundle_approval",
        "pending_approval",
    }
    db_task = db.query(Task).filter(Task.task_id == request.task_id).first()
    if not db_task or db_task.status not in pending_statuses:
        raise HTTPException(status_code=404, detail="Task not found or not pending approval.")

    if not request.approved:
        rejected_step = status_to_step(db_task.status)
        db_task.status = 'rejected'
        db_task.last_rejected_step = rejected_step
        db_task.last_rejected_at = time.strftime("%Y-%m-%d %H:%M:%S")
        db_task.pending_approval_content = None
        db.commit()
        logger.info(f"Task {request.task_id} rejected at step {db_task.last_rejected_step}.")
        return TaskStatus.from_orm(db_task)

    resolved_status = get_next_status(current_status=request.overrides.get('status') if request.overrides else None)
    logger.info(f"Task {request.task_id} approved by human. Resuming graph.")

    # 1. Define the config to resume the correct graph instance
    config = {"configurable": {"thread_id": request.task_id}}
    graph = get_app_graph()

    # 2. Invoke the graph again. The checkpointer loads the state automatically.
    final_state = await graph.ainvoke(None, config)
    logger.info(f"Graph for task {request.task_id} advanced to state '{final_state.get('status')}'.")

    # 3. Update our application DB with the new status/content
    db_task.status = final_state.get('status', db_task.status)
    db_task.pending_approval_content = final_state.get('pending_approval_content')
    artifact_fields = [
        'research_summary',
        'prd_summary',
        'user_stories',
        'user_flow_diagram',
        'wireframe_html',
        'engineering_spec',
        'engineering_file_name',
        'engineering_code',
        'engineering_qa',
    ]
    for field in artifact_fields:
        current_value = getattr(db_task, field)
        new_value = final_state.get(field) if final_state else None
        setattr(db_task, field, current_value or new_value)
    db.commit()
    logger.info(f"Task {request.task_id} updated in DB to final status '{db_task.status}'.")
    
    return TaskStatus.from_orm(db_task)


@app.post("/resubmit_step", response_model=TaskStatus)
async def resubmit_step(request: ResubmitRequest, db: Session = Depends(get_db)):
    allowed_steps = set(STEP_STATUS_MAP.keys())
    if request.step not in allowed_steps:
        raise HTTPException(status_code=400, detail="Invalid step for resubmission.")

    db_task = db.query(Task).filter(Task.task_id == request.task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if db_task.status != "rejected":
        raise HTTPException(status_code=400, detail="Task is not rejected.")
    if db_task.last_rejected_step != request.step:
        raise HTTPException(status_code=400, detail="Resubmit step does not match last rejection.")

    clear_artifacts_for_step(db_task, request.step)
    state = build_state_from_task(db_task)
    state["status"] = STEP_STATUS_MAP[request.step]
    state["pending_approval_content"] = None

    step_fn_map = {
        "research": research_node,
        "product_prd": product_prd_node,
        "product_stories": product_stories_node,
        "ux_design": ux_design_node,
        "engineering": engineering_node,
    }
    step_fn = step_fn_map.get(request.step)
    if not step_fn:
        raise HTTPException(status_code=400, detail="Unsupported step.")

    updated_state = step_fn(state)
    db_task.status = updated_state.get("status", db_task.status)
    db_task.pending_approval_content = updated_state.get("pending_approval_content")
    db_task.research_summary = updated_state.get("research_summary")
    db_task.prd_summary = updated_state.get("prd_summary")
    db_task.user_stories = updated_state.get("user_stories")
    db_task.user_flow_diagram = updated_state.get("user_flow_diagram")
    db_task.wireframe_html = updated_state.get("wireframe_html")
    db_task.engineering_spec = updated_state.get("engineering_spec")
    db_task.engineering_file_name = updated_state.get("engineering_file_name")
    db_task.engineering_code = updated_state.get("engineering_code")
    db_task.engineering_qa = updated_state.get("engineering_qa")
    db_task.last_rejected_step = None
    db_task.last_rejected_at = None
    db.commit()
    db.refresh(db_task)
    logger.info("Task %s resubmitted for step %s", request.task_id, request.step)
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
def get_next_status(current_status: Optional[str]) -> Optional[str]:
    workflow_order = [
        "starting",
        "pending_research_approval",
        "pending_prd_approval",
        "pending_story_approval",
        "pending_ux_approval",
        "pending_engineering_bundle_approval",
        "ready_for_gtm",
        "completed",
        "rejected",
    ]
    if not current_status:
        return None
    try:
        idx = workflow_order.index(current_status)
    except ValueError:
        return current_status
    if idx + 1 < len(workflow_order):
        return workflow_order[idx + 1]
    return current_status
