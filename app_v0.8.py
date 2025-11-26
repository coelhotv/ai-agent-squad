# version 0.8 -- refactoring of UX artifacts rendering on screen

import os
import json
import csv
import re
from io import StringIO
# --- Core Imports ---
import uuid
import logging
import socket
import uvicorn
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
    reasoning_spec = Column(Text, nullable=True)
    engineering_spec = Column(Text, nullable=True)
    engineering_spec_qa = Column(Text, nullable=True)
    reasoning_code = Column(Text, nullable=True)
    engineering_file_name = Column(Text, nullable=True)
    engineering_code = Column(Text, nullable=True)
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
    "reasoning_spec",
    "engineering_spec",
    "engineering_spec_qa",
    "reasoning_code",
    "engineering_file_name",
    "engineering_code",
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
    reasoning_spec: Optional[str]
    engineering_spec: Optional[str]
    engineering_spec_qa: Optional[str]
    reasoning_code: Optional[str]
    engineering_file_name: Optional[str]
    engineering_code: Optional[str]
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
        # "think": reasoning,
    }

    logger.info("Calling Ollama model %s (reasoning=%s).", model, reasoning)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            with httpx.Client(timeout=600) as client: # Increased timeout to 10 minutes
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
            
            content = data.get("response")
            if not content:
                return None

            return _extract_json(content)

        except httpx.TimeoutException as exc:
            logger.warning(f"Ollama request timed out (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt + 1 == max_retries:
                logger.error("Ollama request failed after max retries.")
                return None
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as exc:
            logger.exception("Ollama request failed: %s", exc)
            return None

    return None


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


def format_structured_summary(data: dict) -> str:
    lines = []
    summary = data.get("summary")
    if summary:
        lines.append("## Summary:\n" + summary.strip())

    target_audience = data.get("target_audience")
    if target_audience:
        lines.append("\n## Target Audience:\n" + target_audience.strip())

    opportunities = data.get("opportunities") or []
    if opportunities:
        lines.append("\n## Opportunities:")
        for opp in opportunities:
            title = opp.get("title") or "Opportunity"
            details = opp.get("details") or ""
            lines.append(f"- {title}: {details}")

    risks = data.get("risks") or []
    if risks:
        lines.append("\n## Risks:")
        for risk in risks:
            title = risk.get("title") or "Risk"
            details = risk.get("details") or ""
            lines.append(f"- {title}: {details}")

    refs = data.get("references") or []
    if refs:
        lines.append("\n## References:")
        for ref in refs:
            source = ref.get("source") or "Source"
            url = ref.get("url") or ""
            lines.append(f"- {source}: {url}")

    return "\n".join(lines).strip()


def generate_prd_document(product_idea: str, research_summary: str | None) -> str:
    section_schema = {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["rationale", "items"],
    }

    prd_schema = {
        "type": "object",
        "properties": {
            "app_name": {"type": "string"},
            "executive_summary": {"type": "string"},
            "target_segment": {"type": "string"},
            "market_opportunity": section_schema,
            "customer_needs": section_schema,
            "product_scope": section_schema,
            "launch_milestones": section_schema,
            "critical_metrics": section_schema,
            "dependencies": section_schema,
            "risks": section_schema,
            "success_criteria": section_schema,
        },
        "required": [
            "app_name",
            "executive_summary",
            "target_segment",
            "market_opportunity",
            "customer_needs",
            "product_scope",
            "critical_metrics",
            "success_criteria",
        ]
    }
    system_prompt = (
        "You are a senior PM translating user needs and business opportunities into a "
        "clear, actionable one-page PRD. For every section, explain why it matters, what "
        "outcome we are targeting, and one concrete next step (e.g., a dependency, test, "
        "or measurement). Favor specific examples and metric targets over abstract statements."
    )
    user_prompt = (
        f"Idea: {product_idea}.\nResearch summary: {research_summary or 'n/a'}.\n"
        "Structure the response using the schema: put the most critical insights first, "
        "include a tangible follow-up for each section, and surface actionable metrics, "
        "risks, dependencies, and milestones wherever they help engineering/design take action. "
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
                    "target_segment": {"type": "string"},
                    "market_opportunity": section_schema,
                    "customer_needs": section_schema,
                    "product_scope": section_schema,
                    "launch_milestones": section_schema,
                    "critical_metrics": section_schema,
                    "dependencies": section_schema,
                    "risks": section_schema,
                    "success_criteria": section_schema,
                },
                "required": [
                    "app_name",
                    "executive_summary",
                    "target_segment",
                    "market_opportunity",
                    "customer_needs",
                    "product_scope",
                    "critical_metrics",
                    "success_criteria",
                ],
            },
        }
        parsed = call_perplexity_json(system_prompt, user_prompt, json_schema)
    if not parsed:
        return fallback_prd(product_idea, research_summary)

    lines = ["# App:", parsed.get("app_name", "")]
    lines.append("\n## Executive Summary:")
    lines.append(parsed.get("executive_summary", ""))
    target_segment = parsed.get("target_segment")
    if target_segment:
        lines.append("\n## Target Segment & Context:")
        lines.append(target_segment)

    def append_section(title: str, key: str, max_items: int = 5, required: bool = False) -> None:
        section = parsed.get(key)
        rationale = None
        items: list[str] = []
        if isinstance(section, dict):
            rationale = section.get("rationale")
            items = section.get("items") or []
        else:
            items = section or []
        if not items and not (required or rationale):
            return
        lines.append(f"\n## {title}:")
        if rationale:
            lines.append(f"Rationale: {rationale}")
        if not items:
            lines.append("Details pending.")
            return
        for item in items[:max_items]:
            lines.append(f"- {item}")

    append_section("Market Opportunity & Positioning", "market_opportunity", max_items=3, required=True)
    append_section("Customer Needs", "customer_needs", max_items=4, required=True)
    append_section("Product Scope and Use cases", "product_scope", max_items=4, required=True)
    append_section("Launch Milestones", "launch_milestones", max_items=3)
    append_section("Critical Metrics", "critical_metrics", max_items=3, required=True)
    append_section("Dependencies & Questions", "dependencies", max_items=3)
    append_section("Risks / Unknowns", "risks", max_items=3)
    append_section("Success Criteria", "success_criteria", max_items=3, required=True)
    return "\n".join(lines).strip()


def fallback_prd(product_idea: str, research_summary: str | None) -> str:
    return (
        f"# App:\n{product_idea}\n\n"
        "## Target Segment & Context:\n"
        f"- {research_summary or 'Women 18-45 focused on reproductive health self-care.'}\n\n"
        "## Executive Summary:\n"
        "A first iteration of CycleGuard focuses on a single, high-value workflow that validates "
        "demand while keeping delivery scope tight.\n\n"
        "## Market Opportunity & Positioning:\n"
        "- Based on early research, people managing hormonal health lack proactive cycle insights.\n"
        "- Position as a trusted, AI-informed companion with healthcare-grade safeguards.\n\n"
        "## Customer Needs:\n"
        "- Accurate tracking with minimal manual work.\n"
        "- Symptom logging that reveals meaningful patterns.\n"
        "- Easy visibility into fertility and contraception windows.\n\n"
        "## Product Scope and Use cases:\n"
        "- Cycle tracking with AI-powered predictions.\n"
        "- Health insights that highlight emerging patterns.\n"
        "- Symptom logging dashboard for correlation analysis.\n\n"
        "## Launch Milestones:\n"
        "- Validate primary persona and critical use cases with interviews.\n"
        "- Ship MVP cycle tracker with opt-in symptom logging.\n"
        "- Run closed beta with 20 testers for two full cycles.\n\n"
        "## Critical Metrics:\n"
        "- <draft> 70% of testers complete at least two consecutive cycles.\n"
        "- Prediction accuracy within ±2 days for cycle start.\n"
        "- Privacy NPS > 80% among pilot users.\n\n"
        "## Dependencies & Questions:\n"
        "- Identify trustworthy data sources for cycle prediction modeling.\n"
        "- Legal review of data handling before launch.\n\n"
        "## Risks / Unknowns:\n"
        "- Model drift if users have irregular or postoperative cycles.\n"
        "- Challenge in keeping retention high once novelty wears off.\n\n"
        "## Success Criteria:\n"
        "- 10 pilot sign-ups.\n"
        "- 60% repeat usage in 2 weeks.\n"
        "- Qualitative feedback on usability and privacy."
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
        " Produce two high-value user stories to be in the MVP."
        " For each story, provide 2-3 acceptance criteria."
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

    lines = ["## User Stories:"]
    for i, entry in enumerate(parsed.get("stories", [])[:4], 1):
        story_text = entry.get("story") or ""
        lines.append(f"\n{i}. {story_text}")
        for criteria in entry.get("acceptance_criteria", [])[:3]:
            lines.append(f"  - AC: {criteria}")

    backlog = parsed.get("backlog", [])
    if backlog:
        lines.append("\n## Next Iteration Backlog:")
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


def normalize_mermaid_labels(flow: str) -> str:
    """Normalize Mermaid edges to the `--|Label|` form so Mermaid v10 renders them."""
    import re

    def replace_missing_pipes(match):
        label = match.group("label").strip()
        return f"--|{label}| "

    def replace_full(match):
        label = match.group("label").strip()
        return f"--|{label}|"

    # Convert `-- Yes --` to `--|Yes|`.
    flow = re.sub(r"--\s*(?P<label>[^-|]+?)\s*--", replace_full, flow)
    # Convert `-- Yes|` (missing leading pipe) to `--|Yes|`.
    flow = re.sub(r"--\s*(?P<label>[^|]+?)\|\s*", replace_missing_pipes, flow)
    return flow


def generate_user_flow_diagram(product_idea: str, user_stories: str | None) -> str:
    flow_schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "minItems": 4,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]+$"},
                        "label": {"type": "string"},
                    },
                    "required": ["id", "label"],
                },
                "description": "List 4-6 unique nodes representing the flow."
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["source", "target"],
                },
                "description": "Edges between node IDs; labels are optional but helpful."
            },
            "mermaid_syntax": {
                "type": "string",
                "description": "Optional Mermaid definition mirroring the nodes and edges."
            }
        },
        "required": ["nodes", "edges", "mermaid_syntax"]
    }
    system_prompt = (
        "You are an expert UX Architect. Produce a compact Mermaid flow plus structured node/edge data "
        "that represents the core journey described by the user stories and acceptance criteria.\n"
        "Focus on the MVP path (no login/onboarding) and keep labels short."
    )
    user_prompt = f"""
    Idea: {product_idea}.
    User stories:
    {user_stories or 'Unavailable.'}

    ### GUIDELINES:
     1. Define 4-6 nodes with clean alphanumeric IDs and human-readable labels.
     2. Connect those nodes with edges; include short edge labels when it clarifies transitions.
     3. Mirror this structure in the Mermaid `mermaid_syntax` string (`graph TD` with `A[Label]` style).
     4. Keep every ID/label consistent between the JSON and Mermaid output.
    """
    parsed = call_ollama_json(system_prompt, user_prompt, flow_schema)
    if parsed and isinstance(parsed, dict) and parsed.get("edges"):
        if parsed.get("mermaid_syntax"):
            parsed["mermaid_syntax"] = normalize_mermaid_labels(parsed["mermaid_syntax"])
        return json.dumps(parsed, indent=2)
    return fallback_user_flow_diagram(product_idea)


def fallback_user_flow_diagram(product_idea: str) -> str:
    fallback = {
        "title": f"{product_idea} Flow",
        "nodes": [
            {"id": "Start", "label": "Start"},
            {"id": "Input", "label": "Input Data"},
            {"id": "Review", "label": "Review Data"},
            {"id": "Insights", "label": "View Insights"},
            {"id": "End", "label": "End"},
        ],
        "nodes_list": [
            "Start",
            "Input",
            "Review",
            "Insights",
            "End"
        ],
        "mermaid_syntax": (
            "graph TD\n"
            "  Start[Open App] --> Input[Add Cycle]\n"
            "  Input --> Review{Review Data?}\n"
            "  Review --|Yes| Insights[View Insights]\n"
            "  Review --|No| End[Exit]\n"
        )
    }
    fallback["edges"] = [
        {"source": "Start", "target": "Input", "label": "Enter Data"},
        {"source": "Input", "target": "Review", "label": "Review"},
        {"source": "Review", "target": "Insights", "label": "Yes"},
        {"source": "Review", "target": "End", "label": "No"},
    ]
    fallback["mermaid_syntax"] = normalize_mermaid_labels(fallback["mermaid_syntax"])
    return json.dumps(fallback, indent=2)


def generate_wireframe_html(product_idea: str, user_stories: str | None, flow_diagram: str | None) -> str:
    wireframe_schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Short explanation of how the wireframe satisfies the user stories and flow."
            },
            "html_content": {
                "type": "string",
                "description": "The raw HTML elements. Do NOT include <html>, <head>, or <body> tags."
            }
        },
        "required": ["reasoning", "html_content"]
    }
    system_prompt = (
        "You are a Senior UI/UX Prototyper. Your goal is to create a low-fidelity wireframe writing HTML and Tailwind CSS code. "
        "Deliver the core experience of the MVP product (first user story) inside a low-fi wireframe. "
        "Document your reasoning before emitting the HTML so reviewers understand how the layout maps to the stories."
        "You must use the provided product idea, its user stories + acceptance criteria, and the user flow as input."
    )
    user_prompt = f"""
        Idea: {product_idea}.

        User stories:
        {user_stories or 'Unavailable.'}

        Mermaid user flow:
        {flow_diagram or 'Unavailable.'}

        ## DESIGN RULES:
        - Keep the layout low-fidelity, grayscale/light contrast, and mindful of dark mode.
        - Favor standard components: headers, cards, input sections, and placeholder charts.
        - Avoid external images—use styled div placeholders instead.
        - Use Tailwind utility classes for spacing, typography (`font-bold` for headers), and readability.

        **Output:**
        1. First provide a short reasoning note that ties the layout back to the user stories and flow.
        2. Then output the HTML structure using only divs/sections aligned with the flow.
        3. Do not include <html>, <head>, or <body> tags.
        4. Return the JSON object (reasoning + html_content) only.
    """
    parsed = call_ollama_json(system_prompt, user_prompt, wireframe_schema, reasoning=True)
    if parsed and isinstance(parsed, dict) and parsed.get("html_content"):
        return json.dumps(parsed, indent=2)
    return fallback_wireframe_html(product_idea)


def fallback_wireframe_html(product_idea: str) -> str:
    content = (
        "<div class=\"min-h-screen bg-slate-900 text-slate-100 p-8\">\n"
        "  <header class=\"max-w-4xl mx-auto bg-slate-800 px-6 py-4 rounded-2xl\">\n"
        f"    <p class=\"text-sm uppercase tracking-widest text-slate-400\">Concept</p>\n"
        f"    <h1 class=\"text-2xl font-semibold\">{product_idea}</h1>\n"
        "  </header>\n"
        "  <main class=\"max-w-4xl mx-auto mt-6 grid gap-4 md:grid-cols-2\">\n"
        "    <section class=\"bg-slate-800 rounded-2xl p-4\">\n"
        "      <h2 class=\"text-lg font-medium mb-2\">Logging</h2>\n"
        "      <form class=\"space-y-4\">\n"
        "        <div>\n"
        "          <label class=\"text-sm text-slate-300\" for=\"cigarettes\">Cigarettes smoked</label>\n"
        "          <input id=\"cigarettes\" type=\"number\" class=\"w-full mt-1 rounded-xl px-3 py-2 bg-slate-900 border border-gray-700\" />\n"
        "        </div>\n"
        "        <button class=\"w-full rounded-xl bg-emerald-500 px-4 py-2\">Save Entry</button>\n"
        "      </form>\n"
        "    </section>\n"
        "    <section class=\"bg-slate-800 rounded-2xl p-4\">\n"
        "      <h2 class=\"text-lg font-medium mb-2\">Insights</h2>\n"
        "      <div class=\"h-48 bg-slate-900 rounded-xl border border-gray-700 flex items-center justify-center\">\n"
        "        <span class=\"text-sm text-slate-400\">Chart placeholder</span>\n"
        "      </div>\n"
        "    </section>\n"
        "  </main>\n"
        "</div>\n"
    )
    fallback = {
        "reasoning": "Fallback layout covers logging and insight display with clear hierarchy.",
        "html_content": content
    }
    return json.dumps(fallback, indent=2)


DEMO_BANNED_KEYWORDS = (
    "jwt",
    "oauth",
    "/login",
    "/users/register",
    "password_hash",
    "hashed",
    "persistent database",
    "postgres",
    "mysql",
    "sqlite",
    "production auth",
)


def detect_demo_constraint_violations(steps: list[str]) -> list[str]:
    found: list[str] = []
    normalized = [step.lower() for step in steps]
    for keyword in DEMO_BANNED_KEYWORDS:
        if any(keyword in step for step in normalized):
            found.append(keyword)
    return found


def normalize_ac_id(raw: str | None) -> str:
    if not raw:
        return ""
    normalized = raw.strip()
    if not normalized:
        return ""
    if not normalized.upper().startswith("AC"):
        normalized = f"AC {normalized}"
    return normalized.upper()


def extract_acceptance_criteria(user_stories: str | None) -> set[str]:
    if not user_stories:
        return set()
    matches = re.findall(r"AC\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)*)", user_stories, re.IGNORECASE)
    return {normalize_ac_id(match) for match in matches if normalize_ac_id(match)}


def collect_contract_ac_refs(contract: dict | None) -> set[str]:
    refs: set[str] = set()
    if not contract:
        return refs
    for entry in contract.get("schemas", []) or []:
        for ac in entry.get("ac_refs") or []:
            norm = normalize_ac_id(ac)
            if norm:
                refs.add(norm)
    for entry in contract.get("endpoints", []) or []:
        for ac in entry.get("ac_refs") or []:
            norm = normalize_ac_id(ac)
            if norm:
                refs.add(norm)
    return refs


def format_warning_section(notes: list[str]) -> str:
    if not notes:
        return ""
    lines = ["Warnings/Constraints:"]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


def strip_rationales(text: str | None) -> str | None:
    if not text:
        return text
    cleaned = re.sub(r"(?m)^Rationale:.*(?:\n|$)", "", text)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned.strip())
    return cleaned or None


def append_ac_placeholder(contract: dict[str, list], ac_id: str) -> bool:
    template = AC_PLACEHOLDER_TEMPLATES.get(ac_id)
    if not template:
        return False
    schemas = contract.setdefault("schemas", [])
    endpoints = contract.setdefault("endpoints", [])
    schema = template["schema"].copy()
    if not any(entry["name"] == schema["name"] for entry in schemas):
        schemas.append(schema)
    endpoint = template.get("endpoint")
    if endpoint and not any(entry["path"] == endpoint["path"] and entry["method"] == endpoint["method"] for entry in endpoints):
        endpoints.append(endpoint.copy())
    return True


AC_PLACEHOLDER_TEMPLATES = {
    "AC 1.1": {
        "schema": {
            "name": "CumulativeStatistics",
            "fields": [
                {"name": "total_smoked", "type": "int", "required": True, "validators": []},
                {"name": "average_per_day", "type": "float", "required": True, "validators": []},
            ],
            "ac_refs": ["AC 1.1"],
        },
        "endpoint": {
            "method": "GET",
            "path": "/progress-bar",
            "response_model": "CumulativeStatistics",
            "ac_refs": ["AC 1.1"],
        },
    },
    "AC 1.2": {
        "schema": {
            "name": "SymptomEntry",
            "fields": [
                {"name": "symptoms", "type": "List[str]", "required": True, "validators": []},
                {"name": "notes", "type": "string", "required": False, "validators": []},
            ],
            "ac_refs": ["AC 1.2"],
        },
        "endpoint": {
            "method": "POST",
            "path": "/symptoms",
            "response_model": "SymptomEntry",
            "ac_refs": ["AC 1.2"],
        },
    },
    "AC 1.3": {
        "schema": {
            "name": "CycleCalendarView",
            "fields": [
                {"name": "month", "type": "string", "required": True, "validators": []},
                {"name": "highlighted_dates", "type": "List[string]", "required": True, "validators": []},
            ],
            "ac_refs": ["AC 1.3"],
        },
        "endpoint": {
            "method": "GET",
            "path": "/cycle/calendar",
            "response_model": "CycleCalendarView",
            "ac_refs": ["AC 1.3"],
        },
    },
    "AC 2.1": {
        "schema": {
            "name": "CyclePrediction",
            "fields": [
                {"name": "prediction_date", "type": "date", "required": True, "validators": []},
                {"name": "phase", "type": "string", "required": True, "validators": []},
            ],
            "ac_refs": ["AC 2.1"],
        },
        "endpoint": {
            "method": "POST",
            "path": "/cycle/predict",
            "response_model": "CyclePrediction",
            "ac_refs": ["AC 2.1"],
        },
    },
    "AC 2.2": {
        "schema": {
            "name": "PredictedCycle",
            "fields": [
                {"name": "start_date", "type": "date", "required": True, "validators": []},
                {"name": "end_date", "type": "date", "required": True, "validators": []},
                {"name": "confidence", "type": "float", "required": False, "validators": []},
            ],
            "ac_refs": ["AC 2.2"],
        },
        "endpoint": {
            "method": "GET",
            "path": "/cycle/predictions",
            "response_model": "PredictedCycle",
            "ac_refs": ["AC 2.2"],
        },
    },
    "AC 2.3": {
        "schema": {
            "name": "FertilityInsight",
            "fields": [
                {"name": "likelihood", "type": "string", "required": True, "validators": []},
                {"name": "timeline", "type": "string", "required": True, "validators": []},
            ],
            "ac_refs": ["AC 2.3"],
        },
        "endpoint": {
            "method": "GET",
            "path": "/fertility/insights",
            "response_model": "FertilityInsight",
            "ac_refs": ["AC 2.3"],
        },
    },
}


def generate_engineering_spec(
    product_idea: str,
    user_stories: str | None,
    wireframe_html: str | None,
    qa_feedback: str | None = None,
    reasoning_override: str | None = None,
) -> str:
    # --- Step 1: Generate the reasoning steps with the reasoning model ---
    reasoning_schema = {
        "type": "object",
        "properties": {
            "detailed_steps": {
                "type": "array",
                "description": "Explain your step-by-step plan to meet all user stories and acceptance criteria before defining schemas.",
                "items": {"type": "string"},
            }
        },
        "required": ["detailed_steps"],
    }
    reasoning_system_prompt = (
        "You are a Senior Solutions Architect specializing in FastAPI, Pydantic v2, and uvicorn.\n"
        "Your task is to create a step-by-step detailed plan for a **demo-only prototype API**. This is for a short-lived demo, not a production system.\n"
        "RULES:\n"
        "1. **MVP ONLY**: Your plan must ONLY address the user stories provided. IGNORE all backlog items, future features, or items not in the user stories. Low-fidelity wireframe is provided only for reference.\n"
        "2. **NO PRODUCTION FEATURES**: You are FORBIDDEN from including: real authentication (JWT/OAuth), persistent databases (use in-memory only), background tasks (Celery/cron), or unit tests. Mentioning or designing these will fail the task. No `/login` or `/users/register` endpoints, no hashed passwords, and no third-party auth libraries.\n"
        "3. **SIMULATE, DON'T BUILD**: Assume a single hardcoded user for every call. Any security-related behavior should be described as simulated (e.g., returning a mock token or static consent form). The endpoints can return static or derived data—no scheduling or multi-user logic.\n"
        "4. **DATA FIRST**: Define the Pydantic schemas and their critical fields before any endpoints. Each schema must name the Acceptance Criteria it satisfies (e.g., 'Schema CycleEntry -> AC 1.1').\n"
        "5. **TRACEABILITY**: Each reasoning step must mention the AC numbers it addresses and the schema/endpoint that will deliver it (e.g., 'Step 2 (AC 1.1, AC 1.3): ...').\n"
        "6. **AC COVERAGE**: The final API contract should include only the schemas and endpoints required to meet the listed ACs. Show a short checklist at the end that reruns the rules to self-validate the plan."
        "OUTPUT FORMAT:\n"
        "Output your detailed steps plan only in the provided JSON schema!"
    )
    reasoning_user_prompt = f"""
    Product idea: {product_idea}

    User stories and Acceptance Criteria:
    {user_stories or 'Unavailable.'}

    Wireframe HTML:
    {wireframe_html or 'Unavailable.'}

    {'Previous attempt QA feedback: ' + qa_feedback if qa_feedback else ''}

    When you describe the `detailed_steps`, show how each step satisfies the exact Acceptance Criteria (e.g., 'AC 1.1, AC 1.3') and cite the schema/endpoint that will deliver it. After the reasoning steps, include a short checklist that reconfirms the rules (demo-only, no auth, AC coverage).
    """
    reasoning_parsed = None
    warnings: list[str] = []
    if reasoning_override:
        try:
            parsed_override = json.loads(reasoning_override)
            if isinstance(parsed_override, dict) and parsed_override.get("detailed_steps"):
                reasoning_parsed = parsed_override
        except json.JSONDecodeError:
            logger.warning("Could not parse reasoning_override; falling back to generation.")

    if not reasoning_parsed:
        logger.info("... Generating architect reasoning steps with reasoning model.")
        reasoning_parsed = call_ollama_json(reasoning_system_prompt, reasoning_user_prompt, reasoning_schema, reasoning=True)

    if not reasoning_parsed or not reasoning_parsed.get("detailed_steps"):
        logger.error(f"""... Failed to generate reasoning steps. LLM output: {reasoning_parsed}""")
        return json.dumps({
            "detailed_steps": ["Failed to generate reasoning steps."]
        }, indent=2)

    reasoning_steps = reasoning_parsed["detailed_steps"]
    logger.info("... Architect reasoning steps ready.")
    violations = detect_demo_constraint_violations(reasoning_steps)
    if violations:
        warning = (
            "Detected references to forbidden demo features: "
            f"{', '.join(sorted(set(violations)))}. Please keep the plan limited to simulated behavior."
        )
        logger.warning(warning)
        warnings.append(warning)
        reasoning_steps.append(f"NOTE: {warning}")
    user_ac_set = extract_acceptance_criteria(user_stories)

    # --- Step 2: Generate the schemas and endpoints with the coding model ---
    contract_schema = {
        "type": "object",
        "properties": {
            "schemas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "required": {"type": "boolean"},
                                    "validators": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["name", "type", "required"],
                            },
                        },
                        "ac_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Acceptance criteria this schema fulfills",
                        },
                    },
                    "required": ["name", "fields"],
                },
            },
            "endpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "path": {"type": "string"},
                        "response_model": {"type": "string"},
                        "errors": {"type": "array", "items": {"type": "string"}},
                        "ac_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Acceptance criteria this endpoint satisfies",
                        },
                    },
                    "required": ["method", "path", "response_model"],
                },
            },
        },
        "required": ["schemas", "endpoints"],
    }
    contract_system_prompt = (
        "You are a specialist API contract engineer. Your job is to generate the JSON for API schemas and endpoints based *exactly* on the provided reasoning steps.\n"
        "RULES:\n"
        "1. Implement every step from the reasoning plan.\n"
        "2. Define all Pydantic v2 models in the `schemas` list.\n"
        "3. Define all RESTful endpoints in the `endpoints` list.\n"
        "4. Ensure every model used in an endpoint's `response_model` is defined in `schemas`.\n"
        "5. Every schema and endpoint must include an `ac_refs` array that lists the Acceptance Criteria it satisfies (e.g., 'AC 1.1').\n"
        "6. Output only the JSON structure."
    )
    contract_user_prompt_base = f"""
    Architect's Plan (Reasoning Steps):
    {reasoning_steps}

    {'Previous attempt QA feedback: ' + qa_feedback if qa_feedback else ''}
    
    Now, generate the `schemas` and `endpoints` JSON based on this plan (reasoning steps).
    Each schema and endpoint must list the specific Acceptance Criteria it covers via the `ac_refs` field.
    """
    contract_prompt_suffix = ""
    contract_parsed = None
    retry_count = 0
    max_retries = 1
    while True:
        contract_user_prompt = contract_user_prompt_base + contract_prompt_suffix
        logger.info("... Generating API contract with coding model.")
        contract_parsed = call_ollama_json(contract_system_prompt, contract_user_prompt, contract_schema)
        if not contract_parsed or not contract_parsed.get("schemas") or not contract_parsed.get("endpoints"):
            logger.error("... Failed to generate API contract from reasoning steps.")
            break
        contract_ac_refs = collect_contract_ac_refs(contract_parsed)
        missing_ac = sorted(user_ac_set - contract_ac_refs)
        if not missing_ac:
            break
        added_ac: list[str] = []
        for ac in missing_ac:
            if append_ac_placeholder(contract_parsed, ac):
                added_ac.append(ac)
        if added_ac:
            note = (
                "Automatically added placeholder schemas/endpoints to cover missing ACs: "
                f"{', '.join(sorted(set(added_ac)))}."
            )
            warnings.append(note)
            logger.warning(note)
            contract_ac_refs = collect_contract_ac_refs(contract_parsed)
            missing_ac = sorted(user_ac_set - contract_ac_refs)
            if not missing_ac:
                break
        warning = (
            "Missing AC coverage in the contract: "
            f"{', '.join(missing_ac)}."
        )
        if retry_count < max_retries:
            warning = (
                f"{warning} Re-running the contract generation to cover them."
            )
            warnings.append(warning)
            logger.warning(warning)
            contract_prompt_suffix = (
                f"\n\nPlease cover the following missing Acceptance Criteria: {', '.join(missing_ac)}."
            )
            retry_count += 1
            continue
        warnings.append(warning)
        logger.warning(warning)
        break

    if not contract_parsed or not contract_parsed.get("schemas") or not contract_parsed.get("endpoints"):
        logger.error("... Failed to generate API contract from reasoning steps.")
        return json.dumps({
            "schemas": [],
            "endpoints": [],
            "warnings": warnings,
        }, indent=2)

    # --- Step 3: Combine and return the full spec ---
    full_spec = {
        "schemas": contract_parsed["schemas"],
        "endpoints": contract_parsed["endpoints"],
        "warnings": warnings,
    }

    return json.dumps(full_spec, indent=2)


def run_spec_qa_review(
    product_idea: str,
    user_stories: str | None,
    spec_text: str | None,
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
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor"],
                        },
                        "title": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["severity", "title", "details"],
                },
            },
        },
        "required": ["verdict", "findings"],
    }
    system_prompt = (
        "You are a pragmatic QA Engineer reviewing an API specification for a **demo-only prototype**.\n"
        "Your job is to check if the spec is *sufficient to build a demo* that meets the user stories' acceptance criteria (AC). Do not evaluate it as a production system.\n"
        "RULES:\n"
        "1. **Focus on ACs**: A 'critical' finding is ONLY when an acceptance criterion is impossible to meet with the given spec (e.g., a required field is missing from a schema).\n"
        "2. **Ignore Production Concerns**: DO NOT fail the spec for missing `UPDATE`/`DELETE` endpoints, lack of authentication, or other production-level features. These are out of scope for a demo.\n"
        "3. **Be Actionable**: Findings must point to a specific missing or incorrect part of the spec. Refer to the schemas/endpoints arrays when justifying each AC.\n"
        "4. **Be Concise**: Keep the 'details' for each finding to one or two short sentences.\n"
        "5. **Output JSON**: Your entire response must be in the specified JSON format.\n"
        "6. **Contract Requirements**: Fail the spec if it does not define the necessary schemas and endpoints to satisfy every Acceptance Criterion. The spec must be self-contained and sufficient for a developer to build a working demo."
    )
    spec_contract = {}
    try:
        spec_contract = json.loads(spec_text) if spec_text else {}
    except json.JSONDecodeError:
        spec_contract = {}
    spec_warnings = list(spec_contract.get("warnings") or [])
    contract_ac_refs = collect_contract_ac_refs(spec_contract)
    user_ac_set = extract_acceptance_criteria(user_stories)
    missing_ac = sorted(user_ac_set - contract_ac_refs)
    user_prompt = f"""
    Product idea: {product_idea}
    User stories: 
    {user_stories or 'Unavailable.'}
    Architecture Spec to review:
    {spec_text or 'Unavailable.'}

    {'Warnings from spec: ' + '; '.join(spec_warnings) if spec_warnings else 'Warnings from spec: None.'}

    Strictly review **the spec**. Does it define the necessary schemas and endpoints to satisfy every Acceptance Criterion for a simple demo?
    """
    # Use the REASONING model for this logical review task.
    parsed = call_ollama_json(system_prompt, user_prompt, qa_schema, reasoning=True)
    if not parsed:
        return json.dumps({"verdict": "fail", "findings": [{"severity": "critical", "title": "QA Failure", "details": "Could not generate a structured QA response for the spec."}] }, indent=2)
    verdict = parsed.get("verdict", "unknown")
    findings = parsed.get("findings") or []
    if missing_ac:
        findings.append({
            "severity": "critical",
            "title": "Missing Acceptance Criteria",
            "details": f"The spec lacks references to {', '.join(missing_ac)}."
        })
        verdict = "fail"
    if spec_warnings:
        findings.append({
            "severity": "major",
            "title": "Spec Warnings",
            "details": "; ".join(spec_warnings)
        })
        if verdict != "fail":
            verdict = "major"
    lines = [f"## Verdict: {verdict}"]
    if findings:
        lines.append("## Findings:")
        for finding in findings:
            sev = finding.get("severity") or "info"
            title = finding.get("title") or "Issue"
            details = finding.get("details") or ""
            lines.append(f"- [{sev}] {title}: {details}")
    recs = parsed.get("recommendations") or []
    if recs:
        lines.append("## Recommendations:")
        for rec in recs:
            lines.append(f"- {rec}")
    if findings and verdict != "fail":
        verdict = "fail"
    checklist: list[str] = []
    if missing_ac:
        checklist.append(f"AC coverage missing: {', '.join(missing_ac)}")
    else:
        checklist.append("AC coverage confirmed.")
    if spec_warnings:
        checklist.append("Demo constraints require attention.")
    else:
        checklist.append("Demo constraints appear satisfied.")
    if checklist:
        lines.append("## Checklist:")
        for item in checklist:
            lines.append(f"- {item}")
    lines[0] = f"## Verdict: {verdict}"
    return "\n".join(lines).strip()


def generate_engineering_code(
    product_idea: str,
    user_stories: str | None,
    spec_contract: str | None,
    plan_override: str | None = None,
) -> tuple[str, str]:
    implementation_plan = "# No implementation plan was generated."
    contract_data = {}
    try:
        contract_data = json.loads(spec_contract) if spec_contract else {}
    except json.JSONDecodeError:
        logger.warning("Could not parse spec contract to JSON.")
        contract_data = {}
    spec_warnings = list(contract_data.get("warnings") or [])
    user_ac_set = extract_acceptance_criteria(user_stories)
    contract_ac_refs = collect_contract_ac_refs(contract_data)
    missing_ac = sorted(user_ac_set - contract_ac_refs)
    if missing_ac:
        note = f"Contract missing coverage for ACs: {', '.join(missing_ac)}."
        logger.warning(note)
        spec_warnings.append(note)
    # --- Step 1: Generate implementation plan with the reasoning model ---
    plan_schema = {
        "type": "object",
        "properties": {
            "implementation_plan": {
                "type": "string",
                "description": "Detailed pseudo-code or step-by-step implementation logic for each endpoint. Focus on data flow and in-memory DB interactions."
            }
        },
        "required": ["implementation_plan"]
    }
    plan_system_prompt = (
        "You are a Senior Engineer specialized in algorithm design and your job is to create an implementation plan for a demo app.\n"
        "Your task is to write detailed pseudo-code for each endpoint in the provided API spec.\n"
        "RULES (demo-only):\n"
        "1. Surface the core value prop using hardcoded users, in-memory data, and no production services—this is a short-lived demo.\n"
        "2. For each endpoint, detail the logic: how to handle the request, what data to fetch from the in-memory `DB`, how to process it, and what to return.\n"
        "3. Describe exactly how you will capture detailed event-level entries, persist them across sessions, aggregate them into counters or summaries, and represent any notification simulators mentioned in the spec so live counters and reminders feel functional for the single persona.\n"
        "4. Show how notification simulators will queue reminders, toggle delivery state, or log retries so push behavior can be inspected without real providers.\n"
        "5. Reference every relevant Acceptance Criterion within the steps (e.g., 'Step 2 (AC 1.2): ...').\n"
        "6. Provide at least three numbered steps and explain how each step maps to the data models or endpoints.\n"
        "7. Plan for an in-memory database (a Python dictionary `DB = {}`). Specify how you will structure and access data (e.g., `DB['users']`, `DB['sessions']`).\n"
        "8. Simulate any security/privacy behavior: use mock tokens, encrypted blobs, or consent placeholders and never install real auth/encryption libraries."
    )
    warnings_block = f"\n{format_warning_section(spec_warnings)}" if spec_warnings else ""
    plan_base_prompt = f"""
    Product Idea (for context): {product_idea}
    User Stories (for business logic context): {user_stories or 'N/A'}
    API Specification (source of truth):
    {spec_contract or 'Unavailable.'}

    Based on the API specification and user stories, provide a detailed implementation plan in pseudo-code.
    {warnings_block}
    """
    implementation_plan = implementation_plan  # start with default text
    plan_parsed = None
    if plan_override:
        try:
            parsed_override = json.loads(plan_override)
            if isinstance(parsed_override, dict) and parsed_override.get("implementation_plan"):
                implementation_plan = parsed_override["implementation_plan"]
            else:
                implementation_plan = plan_override
        except json.JSONDecodeError:
            logger.warning("Could not parse plan_override as JSON; using raw override content.")
            implementation_plan = plan_override

    if not implementation_plan or implementation_plan.strip().startswith("# Code generation failed"):
        logger.info("... Generating implementation plan with reasoning model.")
        plan_parsed = call_ollama_json(plan_system_prompt, plan_base_prompt, plan_schema, reasoning=True)
        if plan_parsed and plan_parsed.get("implementation_plan"):
            implementation_plan = plan_parsed["implementation_plan"]
            logger.info("... Implementation plan generated successfully.")
        else:
            logger.error("... Failed to generate implementation plan.")
            return "main.py", "# Code generation failed: Could not create an implementation plan."

    if plan_parsed and plan_parsed.get("implementation_plan"):
        implementation_plan = plan_parsed["implementation_plan"]
    plan_violations = detect_demo_constraint_violations([implementation_plan])
    if plan_violations:
        violation_note = (
            "Implementation plan mentions forbidden demo features: "
            f"{', '.join(sorted(set(plan_violations)))}."
        )
        logger.warning(violation_note)
        spec_warnings.append(violation_note)

    # --- Step 2: Generate the final code with the coding model ---
    code_schema = {
        "type": "object",
        "properties": {
            "file_name": { "type": "string" },
            "code": { "type": "string", "description": "Complete, runnable Python code." }
        },
        "required": ["file_name", "code"]
    }
    code_system_prompt = (
        "You are an experienced Software Engineer specialized in Python coding. Your task is to translate the provided implementation plan and API spec into a single, runnable FastAPI file for a demo-only app.\n"
        "## RULES (demo-only):\n"
        "1. **Strict Adherence**: Follow the implementation plan and API spec *exactly*. Do not add any logic, endpoints, or models not present in the plan or spec.\n"
        "2. **Single File**: The entire application must be in one Python file.\n"
        "3. **In-Memory DB**: Use a simple dictionary `DB = {}` for storage as outlined in the plan.\n"
        "4. **Complete Code**: The code must be complete, runnable, and include all necessary imports (FastAPI, Pydantic, etc.) and a `uvicorn` runner block.\n"
        "5. **Simulated Behavior**: Build the code so it stores granular event entries, computes counters or aggregates, and surfaces simulated notification behavior (e.g., queue reminders or toggle delivery flags) while keeping the demo tied to a fixed persona.\n"
        "6. **Output Format**: Return only the raw JSON with `file_name` and `code`.\n"
        "7. **Demo Constraints**: Honor the warnings from the spec, keep security behavior simulated, and avoid mentioning banned keywords (jwt, oauth, password_hash, persistent database, etc.)."
    )
    code_warnings_block = f"\n{format_warning_section(spec_warnings)}" if spec_warnings else ""
    code_user_prompt = f"""
    API Specification:
    {spec_contract or 'Unavailable.'}

    Implementation Plan:
    {implementation_plan}

    Now, write the complete Python code for `main.py` based on the provided spec and plan.
    {code_warnings_block}
    """
    logger.info("... Generating Python code from implementation plan with coding model.")
    code_parsed = call_ollama_json(code_system_prompt, code_user_prompt, code_schema)

    if code_parsed and code_parsed.get("file_name") and code_parsed.get("code"):
        file_name = code_parsed["file_name"]
        code = code_parsed["code"]
        # Return the file name and the JSON string
        return file_name, code

    logger.error("... Failed to generate Python code from implementation plan.")
    return "main.py", "# Code generation failed."


def run_engineering_qa_review(
    product_idea: str,
    user_stories: str | None,
    spec_contract: str | None,
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
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor"],
                        },
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
        "You are a QA Engineer reviewing a **demo-only prototype** FastAPI app. Your goal is to ensure the code correctly implements the provided spec for a simple demo.\n"
        "RULES:\n"
        "1. **Primary Goal**: Your review must answer one question: Does the provided code **exactly** implement the schemas and endpoints from the `Architecture Spec`? A 'fail' verdict is required if there is any deviation.\n"
        "2. **Prototype Scope**: The code *should* use in-memory storage (like a dictionary) and have no real authentication. DO NOT flag these as issues.\n"
        "3. **Out of Scope**: You are FORBIDDEN from mentioning features or requirements from the 'Next Iteration Backlog' section of the user stories. Your review is for the MVP only.\n"
        "4. **Be Concise**: Keep findings brief and actionable. Keep the 'details' for each finding to one or two short sentences.\n"
        "5. **Output JSON**: Your entire response must be in the specified JSON format.\n"
        "6. **Verify Business Logic**: Check that functions and endpoints contain a plausible, simple implementation of the business logic required by the user stories' acceptance criteria. Flag empty or placeholder logic as a 'critical' finding.\n"
        "7. **Respect Warnings**: Honor any warnings from the spec, keep security behavior simulated, and do not mention banned keywords (jwt, oauth, password_hash, persistent database, etc.)."
    )
    contract_data = {}
    try:
        contract_data = json.loads(spec_contract) if spec_contract else {}
    except json.JSONDecodeError:
        contract_data = {}
    spec_warnings = list(contract_data.get("warnings") or [])
    contract_ac_refs = collect_contract_ac_refs(contract_data)
    user_ac_set = extract_acceptance_criteria(user_stories)
    missing_ac = sorted(user_ac_set - contract_ac_refs)
    code_violations = detect_demo_constraint_violations([engineering_code or ""])
    user_prompt = f"""
    Product idea: {product_idea}
    User stories: {user_stories or 'Unavailable.'}
    Architetural Spec: {spec_contract or 'Unavailable.'}

    Prototype code to review:
    {engineering_code or 'Unavailable.'}
    
    {'Warnings from spec: ' + '; '.join(spec_warnings) if spec_warnings else 'Warnings from spec: None.'}
    """
    parsed = call_ollama_json(system_prompt, user_prompt, qa_schema, reasoning=True)
    if not parsed:
        return json.dumps({"verdict": "fail", "findings": [{"severity": "critical", "title": "QA Failure", "details": "Could not generate a structured QA response for the spec."}] }, indent=2)
    verdict = parsed.get("verdict", "unknown")
    findings = parsed.get("findings") or []
    if missing_ac:
        findings.append({
            "severity": "critical",
            "title": "Missing Acceptance Criteria",
            "details": f"The spec/code lacks references to {', '.join(missing_ac)}."
        })
        verdict = "fail"
    if spec_warnings:
        findings.append({
            "severity": "major",
            "title": "Spec Warnings",
            "details": "; ".join(spec_warnings)
        })
        if verdict != "fail":
            verdict = "major"
    if code_violations:
        findings.append({
            "severity": "major",
            "title": "Demo Constraint Violations in Code",
            "details": f"The code mentions: {', '.join(sorted(set(code_violations)))}."
        })
        verdict = "fail"
    lines = [f"## Verdict: {verdict}"]
    if findings:
        lines.append("## Findings:")
        for finding in findings:
            sev = finding.get("severity") or "info"
            title = finding.get("title") or "Issue"
            details = finding.get("details") or ""
            lines.append(f"- [{sev}] {title}: {details}")
    recs = parsed.get("recommendations") or []
    if recs:
        lines.append("## Recommendations:")
        for rec in recs:
            lines.append(f"- {rec}")
    checklist: list[str] = []
    if missing_ac:
        checklist.append(f"AC coverage missing: {', '.join(missing_ac)}")
    else:
        checklist.append("AC coverage confirmed.")
    if spec_warnings:
        checklist.append("Demo spec warnings need attention.")
    else:
        checklist.append("Demo spec warnings cleared.")
    if code_violations:
        checklist.append("Code violates demo constraints.")
    else:
        checklist.append("Code respects demo constraints.")
    if checklist:
        lines.append("## Checklist:")
        for item in checklist:
            lines.append(f"- {item}")
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
    prd_without_rationale = strip_rationales(state.get('prd_summary'))
    state['user_stories'] = generate_user_stories(
        state['product_idea'], prd_without_rationale
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


def engineering_spec_reasoning_node(state: AgentState):
    logger.info(f"--- Node: engineering_spec_reasoning_node (Task ID: {state['task_id']}) ---")
    reasoning_schema = {
        "type": "object",
        "properties": {
            "detailed_steps": {
                "type": "array",
                "description": "Explain your step-by-step plan to meet all user stories and acceptance criteria before defining schemas.",
                "items": {"type": "string"},
            }
        },
        "required": ["detailed_steps"],
    }
    reasoning_system_prompt = (
        "You are a Senior Software Architect specializing in FastAPI, Pydantic v2, and uvicorn.\n"
        "Your task is to create a step-by-step detailed plan for a **demo-only prototype API**. This is for a short-lived demo, not a production system.\n"
        "RULES:\n"
        "1. **MVP ONLY**: Your plan must ONLY address the user stories provided. IGNORE all backlog items, future features, or items not in the user stories. Low-fidelity wireframe is provided only for reference.\n"
        "2. **NO PRODUCTION FEATURES**: You are FORBIDDEN from including: real authentication (JWT/OAuth), persistent databases (use in-memory only), background tasks (Celery/cron), or unit tests. Mentioning or designing these will fail the task. No `/login` or `/users/register` endpoints, no hashed passwords, and no third-party auth libraries.\n"
        "3. **SIMULATE, DON'T BUILD**: Assume a single hardcoded user for every call. Any security-related behavior should be described as simulated (e.g., returning a mock token or static consent form). The endpoints can return static or derived data—no scheduling or multi-user logic.\n"
        "4. **DATA FIRST**: Define the Pydantic schemas and their critical fields before any endpoints. Each schema must name the Acceptance Criteria it satisfies (e.g., 'Schema CycleEntry -> AC 1.1').\n"
        "5. **TRACEABILITY**: Each reasoning step must mention the AC numbers it addresses and the schema/endpoint that will deliver it (e.g., 'Step 2 (AC 1.1, AC 1.3): ...').\n"
        "6. **AC COVERAGE**: The final API contract should include only the schemas and endpoints required to meet the listed ACs. Show a short checklist at the end that reruns the rules to self-validate the plan."
        "OUTPUT FORMAT:\n"
        "Output your detailed steps plan only in the provided JSON schema!"
    )
    reasoning_user_prompt = f"""
    Product idea: {state['product_idea']}

    User stories and Acceptance Criteria:
    {state.get('user_stories') or 'Unavailable.'}

    Wireframe HTML:
    {state.get('wireframe_html') or 'Unavailable.'}
    """
    reasoning_parsed = call_ollama_json(reasoning_system_prompt, reasoning_user_prompt, reasoning_schema, reasoning=True)
    if not reasoning_parsed or not reasoning_parsed.get("detailed_steps"):
        logger.error("... Failed to generate spec reasoning.")
        reasoning_artifact = json.dumps({"detailed_steps": ["Failed to generate reasoning steps."]}, indent=2)
    else:
        reasoning_artifact = json.dumps(reasoning_parsed, indent=2)
    state['reasoning_spec'] = reasoning_artifact
    state['status'] = "pending_spec_reasoning_approval"
    state['pending_approval_content'] = (
        "Architect reasoning is ready. Review the reasoning steps and approve to generate the API contract."
    )
    return state


def engineering_spec_node(state: AgentState):
    logger.info(f"--- Node: engineering_spec_node (Task ID: {state['task_id']}) ---")
    logger.info("... Running architect to generate spec.")
    spec = generate_engineering_spec(
        state['product_idea'],
        state.get('user_stories'),
        state.get('wireframe_html'),
        reasoning_override=state.get('reasoning_spec'),
        # No longer passing QA feedback
    )
    state['engineering_spec'] = spec

    logger.info("... Running QA to review spec for human review.")
    if spec:
        qa_review_json = run_spec_qa_review(
            state['product_idea'],
            state.get('user_stories'),
            spec,
        )
        state['engineering_spec_qa'] = qa_review_json
    else:
        logger.warning("Could not QA spec contract. Please reject and re-submit the stage.")
        state['engineering_spec_qa'] = json.dumps({"verdict": "fail", "findings": [{"severity": "critical", "title": "QA Failure", "details": "Spec was unavailable for QA."}] }, indent=2)

    state['status'] = "pending_spec_approval"
    state['pending_approval_content'] = (
        "Architect has produced an API specification. "
        "Review the spec and the QA report, then approve to proceed to implementation."
    )

    return state


def engineering_code_reasoning_node(state: AgentState):
    logger.info(f"--- Node: engineering_code_reasoning_node (Task ID: {state['task_id']}) ---")
    plan_schema = {
        "type": "object",
        "properties": {
            "implementation_plan": {
                "type": "string",
                "description": "Detailed pseudo-code or step-by-step implementation logic for each endpoint. Focus on data flow and in-memory DB interactions."
            }
        },
        "required": ["implementation_plan"]
    }
    plan_system_prompt = (
        "You are a Senior Engineer specialized in coding algorithms and will create an implementation plan for a demo app.\n"
        "Your task is to write detailed pseudo-code that covers each and all endpoints in the provided API spec.\n"
        "RULES (demo-only):\n"
        "1. Surface the core value prop using hardcoded users, in-memory data, and no production services—this is a short-lived demo.\n"
        "2. For each endpoint, detail the logic: how to handle the request, what data to fetch from the in-memory `DB`, how to process it, and what to return.\n"
        "3. Describe exactly how you will capture detailed event-level entries, persist them across sessions, and aggregate them into counters or summaries so live views remain consistent for the single persona.\n"
        "4. Show how notification simulators will queue reminders, toggle delivery state, or log retries so push behavior can be inspected without real providers.\n"
        "5. Reference every relevant Acceptance Criterion within the steps (e.g., 'Step 2 (AC 1.2): ...').\n"
        "6. Provide at least three numbered steps and explain how each step maps to the data models or endpoints.\n"
        "7. Plan for an in-memory database (a Python dictionary `DB = {}`). Specify how you will structure and access data (e.g., `DB['users']`, `DB['sessions']`).\n"
        "8. Simulate any security/privacy behavior: use mock tokens, encrypted blobs, or consent placeholders and never install real auth/encryption libraries."
    )
    plan_user_prompt = f"""
    Product Idea (for context): {state['product_idea']}
    User Stories (for business logic context): {state.get('user_stories') or 'N/A'}
    API Reasoning Steps (for architectural context): {state.get('reasoning_spec') or 'Unavailable.'}
    
    API Specification (source of truth):
    {state.get('engineering_spec') or 'Unavailable.'}

    Based on the API specification and product's user stories, provide a detailed implementation plan for all spec schema in pseudo-code.
    """
    plan_parsed = call_ollama_json(plan_system_prompt, plan_user_prompt, plan_schema, reasoning=True)
    if not plan_parsed or not plan_parsed.get("implementation_plan"):
        logger.error("... Failed to generate implementation plan reasoning.")
        plan_artifact = json.dumps({"implementation_plan": "# Failed to generate implementation plan."}, indent=2)
    else:
        plan_artifact = json.dumps(plan_parsed, indent=2)
    state["reasoning_code"] = plan_artifact
    state["status"] = "pending_code_reasoning_approval"
    state["pending_approval_content"] = (
        "Developer reasoning is ready. Review the implementation plan and approve to generate the code."
    )
    return state


def engineering_node(state: AgentState):
    logger.info(f"--- Node: engineering_node (Task ID: {state['task_id']}) ---")

    # Pass the full engineering spec (including warnings) to downstream agents so they can honor constraints.
    file_name, code = generate_engineering_code(
        state['product_idea'],
        state.get('user_stories'),
        state.get('engineering_spec'),
        plan_override=state.get('reasoning_code'),
    )
    state['engineering_file_name'] = file_name or "main.py"
    state['engineering_code'] = code or "# Code generation failed"

    logger.info("... Running QA to review code")
    qa_review = run_engineering_qa_review(
        state['product_idea'],
        state.get('user_stories'),
        state.get('engineering_spec'),
        code,
    )
    state['engineering_qa'] = qa_review or "QA review failed"

    state['status'] = "pending_code_approval"
    state['pending_approval_content'] = (
        "Developer has implemented the spec. Review the code and approve to complete."
    )
    return state


def approved_node(state: AgentState):
    logger.info(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "completed"
    state['pending_approval_content'] = None
    return state


# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("product_prd", product_prd_node)
workflow.add_node("product_stories", product_stories_node)
workflow.add_node("ux_design", ux_design_node)
workflow.add_node("engineering_spec_reasoning", engineering_spec_reasoning_node)
workflow.add_node("engineering_spec", engineering_spec_node)
workflow.add_node("engineering_code_reasoning", engineering_code_reasoning_node)
workflow.add_node("engineering", engineering_node)
workflow.add_node("approved", approved_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "product_prd")
workflow.add_edge("product_prd", "product_stories")
workflow.add_edge("product_stories", "ux_design")
workflow.add_edge("ux_design", "engineering_spec_reasoning")
workflow.add_edge("engineering_spec_reasoning", "engineering_spec")
workflow.add_edge("engineering_spec", "engineering_code_reasoning")
workflow.add_edge("engineering_code_reasoning", "engineering")
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
            "engineering_spec_reasoning",
            "engineering_spec",
            "engineering_code_reasoning",
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
    reasoning_spec: Optional[str] = None
    engineering_spec: Optional[str] = None
    engineering_spec_qa: Optional[str] = None
    reasoning_code: Optional[str] = None
    engineering_file_name: Optional[str] = None
    engineering_code: Optional[str] = None
    engineering_qa: Optional[str] = None
    last_rejected_step: Optional[str] = None
    last_rejected_at: Optional[str] = None
    class Config:
        from_attributes = True


class ArtifactUpdateRequest(BaseModel):
    task_id: str
    artifact_name: str
    content: str


class RespondToApprovalRequest(BaseModel):
    task_id: str
    approved: bool
    overrides: dict[str, str] | None = None


class ResubmitRequest(BaseModel):
    task_id: str
    step: str  # e.g., research, product_prd, product_stories, ux_design, engineering_spec, engineering_code


PENDING_STATUSES = {
    "pending_research_approval",
    "pending_prd_approval",
    "pending_story_approval",
    "pending_ux_approval",
    "pending_spec_reasoning_approval",
    "pending_spec_approval",
    "pending_code_reasoning_approval",
    "pending_code_approval",
    "pending_approval",
}


STEP_STATUS_MAP = {
    "research": "pending_research_approval",
    "product_prd": "pending_prd_approval",
    "product_stories": "pending_story_approval",
    "ux_design": "pending_ux_approval",
    "engineering_spec_reasoning": "pending_spec_reasoning_approval",
    "engineering_spec": "pending_spec_approval",
    "engineering_code_reasoning": "pending_code_reasoning_approval",
    "engineering_code": "pending_code_approval",
}


def clear_artifacts_for_step(db_task: Task, step: str):
    """Drop artifacts at and after the specified step so regeneration is clean."""
    downstream_fields: dict[str, list[str]] = {
        "research": [
            "research_summary",
            "prd_summary",
            "user_stories",
            "user_flow_diagram",
            "wireframe_html",
            "reasoning_spec",
            "engineering_spec",
            "engineering_spec_qa",
            "reasoning_code",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "product_prd": [
            "prd_summary",
            "user_stories",
            "user_flow_diagram",
            "wireframe_html",
            "reasoning_spec",
            "engineering_spec",
            "engineering_spec_qa",
            "reasoning_code",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "product_stories": [
            "user_stories",
            "user_flow_diagram",
            "wireframe_html",
            "reasoning_spec",
            "engineering_spec",
            "engineering_spec_qa",
            "reasoning_code",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "ux_design": [
            "user_flow_diagram",
            "wireframe_html",
            "engineering_file_name",
            "reasoning_spec",
            "engineering_spec",
            "engineering_spec_qa",
            "reasoning_code",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "engineering_spec_reasoning": [
            # keep reasoning_spec intact so downstream reuse works
            "engineering_spec",
            "engineering_spec_qa",
            "reasoning_code",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "engineering_spec": [
            "engineering_spec",
            "engineering_spec_qa",
            "reasoning_code",
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "engineering_code_reasoning": [
            # keep reasoning_code intact so downstream reuse works
            "engineering_file_name",
            "engineering_code",
            "engineering_qa",
        ],
        "engineering_code": [
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
        "pending_spec_reasoning_approval": "engineering_spec_reasoning",
        "pending_spec_approval": "engineering_spec",
        "pending_code_reasoning_approval": "engineering_code_reasoning",
        "pending_code_approval": "engineering_code",
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
        "reasoning_spec": db_task.reasoning_spec,
        "engineering_spec": db_task.engineering_spec,
        "engineering_spec_qa": db_task.engineering_spec_qa,
        "reasoning_code": db_task.reasoning_code,
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
        reasoning_spec=None,
        engineering_spec=None,
        engineering_spec_qa=None,
        reasoning_code=None,
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
    # logger.info(f"Current graph state for task {task_id}: {interrupted_state.values}")

    # 5. Update our application DB with the new status from the graph
    db_task.status = interrupted_state.values['status']
    db_task.pending_approval_content = interrupted_state.values['pending_approval_content']
    db_task.research_summary = interrupted_state.values.get('research_summary')
    db_task.prd_summary = interrupted_state.values.get('prd_summary')
    db_task.user_stories = interrupted_state.values.get('user_stories')
    db_task.user_flow_diagram = interrupted_state.values.get('user_flow_diagram')
    db_task.wireframe_html = interrupted_state.values.get('wireframe_html')
    db_task.reasoning_spec = interrupted_state.values.get('resoning_spec')
    db_task.engineering_spec = interrupted_state.values.get('engineering_spec')
    db_task.engineering_spec_qa = interrupted_state.values.get('engineering_spec_qa')
    db_task.reasoning_code = interrupted_state.values.get('resoning_code')
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
        "pending_spec_reasoning_approval",
        "pending_spec_approval",
        "pending_code_reasoning_approval",
        "pending_code_approval",
        "pending_approval",
    ]
    pending_task = (
        db.query(Task)
        .filter(Task.status.in_(pending_statuses))
        .order_by(Task.task_id.desc())
        .first()
    )
    if pending_task:
        return TaskStatus.from_orm(pending_task)
    return None


@app.get("/tasks", response_model=List[TaskStatus])
def list_tasks(db: Session = Depends(get_db)):
    tasks = db.query(Task).order_by(Task.task_id.desc()).all()
    return [TaskStatus.from_orm(task) for task in tasks]


@app.post("/update_artifact", response_model=TaskStatus)
async def update_artifact(request: ArtifactUpdateRequest, db: Session = Depends(get_db)):
    db_task = db.query(Task).filter(Task.task_id == request.task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if db_task.status not in PENDING_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Artifacts can only be edited while the task is pending approval.",
        )

    if not hasattr(db_task, request.artifact_name):
        raise HTTPException(status_code=400, detail=f"Invalid artifact name: {request.artifact_name}")

    # 1. Update the main database
    setattr(db_task, request.artifact_name, request.content)
    db.commit()
    db.refresh(db_task)
    logger.info(f"Updated artifact '{request.artifact_name}' for task {request.task_id} in DB.")

    # 2. Update the graph's checkpointed state
    config = {"configurable": {"thread_id": request.task_id}}
    await get_app_graph().aupdate_state(config, {request.artifact_name: request.content})
    logger.info(f"Updated artifact '{request.artifact_name}' for task {request.task_id} in LangGraph state.")

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
            "Reasoning Spec",
            "Engineering Spec",
            "Engineering Spec QA",
            "Reasoning Code",
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
                task.reasoning_spec or "",
                task.engineering_spec or "",
                task.engineering_spec_qa or "",
                task.reasoning_code or "",
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
        "pending_spec_reasoning_approval",
        "pending_spec_approval",
        "pending_code_reasoning_approval",
        "pending_code_approval",
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

    logger.info(f"Task {request.task_id} approved by human. Resuming graph.")

    # Immediately update the status to 'processing' to give the user feedback
    db_task.status = "processing"
    db_task.pending_approval_content = "Approved. The next agent is now working on the task..."
    db.commit()
    db.refresh(db_task)

    # 1. Apply any overrides from the user before resuming the graph
    config = {"configurable": {"thread_id": request.task_id}}
    if request.overrides:
        logger.info(f"Applying overrides for task {request.task_id}: {list(request.overrides.keys())}")
        # Persist overrides to the main DB as well
        for field, value in request.overrides.items():
            if hasattr(db_task, field):
                setattr(db_task, field, value)
        await get_app_graph().aupdate_state(config, request.overrides)
    # Ensure previously approved reasoning artifacts are preserved when resuming downstream steps
    carry_over: dict[str, str] = {}
    if db_task.reasoning_spec:
        carry_over["reasoning_spec"] = db_task.reasoning_spec
    if db_task.reasoning_code:
        carry_over["reasoning_code"] = db_task.reasoning_code
    if carry_over:
        await get_app_graph().aupdate_state(config, carry_over)

    # 2. Define the config to resume the correct graph instance
    graph = get_app_graph()
    
    # 3. Invoke the graph again. The checkpointer loads the state automatically.
    final_state = await graph.ainvoke(None, config)
    logger.info(f"Graph for task {request.task_id} advanced to state '{final_state.get('status')}'.")

    # 4. Update our application DB with the new status/content
    db_task.status = final_state.get('status', db_task.status)
    db_task.pending_approval_content = final_state.get('pending_approval_content')
    
    artifact_fields = [
        'research_summary',
        'prd_summary',
        'user_stories',
        'user_flow_diagram',
        'wireframe_html',
        'reasoning_spec',
        'engineering_spec',
        'engineering_spec_qa',
        'reasoning_code',
        'engineering_file_name',
        'engineering_code',
        'engineering_qa',
    ]
    for field in artifact_fields:
        if final_state and final_state.get(field):
            setattr(db_task, field, final_state.get(field))

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

    # This logic is based on the original, working implementation from app_old.py
    # It directly calls the node function instead of manipulating the graph state.
    logger.info(f"Task {request.step} resubmitted for a re-run")

    clear_artifacts_for_step(db_task, request.step)
    state = build_state_from_task(db_task)
    state["status"] = STEP_STATUS_MAP[request.step]
    state["pending_approval_content"] = None

    step_fn_map = {
        "research": research_node,
        "product_prd": product_prd_node,
        "product_stories": product_stories_node,
        "ux_design": ux_design_node,
        "engineering_spec_reasoning": engineering_spec_reasoning_node,
        "engineering_spec": engineering_spec_node,
        "engineering_code_reasoning": engineering_code_reasoning_node,
        "engineering_code": engineering_node,
    }
    step_fn = step_fn_map.get(request.step)
    if not step_fn:
        raise HTTPException(status_code=400, detail="Unsupported step for resubmission.")

    # Manually run the agent node to regenerate artifacts
    updated_state = step_fn(state)

    # Update the database with the new state from the direct node call
    db_task.status = updated_state.get("status", db_task.status)
    db_task.pending_approval_content = updated_state.get("pending_approval_content")
    artifact_fields = list(AgentState.__annotations__.keys())
    for field in artifact_fields:
        if field in ['task_id', 'product_idea', 'status', 'pending_approval_content']:
            continue
        if updated_state.get(field):
            setattr(db_task, field, updated_state.get(field))

    db_task.last_rejected_step = None
    db_task.last_rejected_at = None
    db.commit()
    db.refresh(db_task)
    
    logger.info("Task %s resubmitted for step %s, status is now %s", request.task_id, request.step, db_task.status)
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
        "pending_spec_reasoning_approval",
        "pending_spec_approval",
        "pending_code_reasoning_approval",
        "pending_code_approval",
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


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    if settings.run_locally:
        host = "0.0.0.0"
        # When reload is True, uvicorn needs an import string to be able to re-import the app.
        uvicorn.run("app:app", host=host, port=port, reload=True)
    else:
        # When not reloading, we can pass the app object directly.
        uvicorn.run(app, host=host, port=port, reload=False)
