# --- Core Imports ---
import uuid
import logging
from contextlib import AsyncExitStack

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
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
from duckduckgo_search import DDGS
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# --- Application Database (SQLite) ---
# This DB stores the high-level status of tasks for the UI
DATABASE_URL = "sqlite:////data/tasks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(String, primary_key=True, index=True)
    product_idea = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    pending_approval_content = Column(Text, nullable=True)
    research_summary = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

def ensure_research_column():
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("tasks")]
    if "research_summary" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN research_summary TEXT"))

ensure_research_column()

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

# Keep the async connection open for the lifetime of the app and close on shutdown.
_checkpointer_stack = AsyncExitStack()
memory_saver: Optional[AsyncSqliteSaver] = None
app_graph = None

def run_research_query(product_idea: str) -> str:
    """Run a lightweight DuckDuckGo search and summarize the top findings."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(product_idea, max_results=3))
    except Exception as exc:
        logger.exception("Research query failed for idea '%s'", product_idea)
        return f"Research unavailable due to error: {exc}"

    if not results:
        return "No public findings were returned for this idea."

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

# --- Agent Node Functions ---
# These nodes now only focus on modifying the state dictionary.
# The main application logic will handle database updates.
def research_node(state: AgentState):
    logger.info(f"--- Node: research_node (Task ID: {state['task_id']}) ---")
    state['research_summary'] = run_research_query(state['product_idea'])
    state['status'] = "research_completed"
    return state

def intake_node(state: AgentState):
    logger.info(f"--- Node: intake_node (Task ID: {state['task_id']}) ---")
    state['status'] = "pending_approval"
    summary = state.get('research_summary') or "Research summary unavailable."
    state['pending_approval_content'] = (
        f"Research findings for '{state['product_idea']}':\n\n{summary}\n\nApprove to proceed?"
    )
    return state

def approved_node(state: AgentState):
    logger.info(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "completed"
    return state

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("intake", intake_node)
workflow.add_node("approved", approved_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "intake")
workflow.add_edge("intake", "approved")
workflow.add_edge("approved", END)


async def initialize_graph():
    """Initialize the LangGraph checkpointer and compiled workflow."""
    global memory_saver, app_graph
    memory_saver = await _checkpointer_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string("/data/checkpoints.sqlite")
    )
    app_graph = workflow.compile(
        checkpointer=memory_saver,
        interrupt_before=["approved"]
    )


def get_app_graph():
    if app_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not initialized.")
    return app_graph

# --- FastAPI Application ---
app = FastAPI(title="Multi-Agent Product Squad API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
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
    db.commit()
    db.refresh(db_task)
    logger.info(f"Task {task_id} updated in DB to status '{db_task.status}'.")

    return TaskStatus.from_orm(db_task)

@app.get("/get_pending_approval", response_model=Optional[TaskStatus])
def get_pending_approval(db: Session = Depends(get_db)):
    pending_task = db.query(Task).filter(Task.status == "pending_approval").first()
    if pending_task:
        return TaskStatus.from_orm(pending_task)
    return None

@app.get("/tasks", response_model=List[TaskStatus])
def list_tasks(db: Session = Depends(get_db)):
    tasks = db.query(Task).order_by(Task.task_id.desc()).all()
    return [TaskStatus.from_orm(task) for task in tasks]

@app.post("/respond_to_approval", response_model=TaskStatus)
async def respond_to_approval(request: RespondToApprovalRequest, db: Session = Depends(get_db)):
    db_task = db.query(Task).filter(Task.task_id == request.task_id).first()
    if not db_task or db_task.status != 'pending_approval':
        raise HTTPException(status_code=404, detail="Task not found or not pending approval.")

    if not request.approved:
        db_task.status = 'rejected'
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
    logger.info(f"Graph for task {request.task_id} finished execution.")

    # 3. Update our application DB with the final status
    db_task.status = final_state.get('status', 'completed')
    db_task.pending_approval_content = None
    db_task.research_summary = final_state.get('research_summary', db_task.research_summary)
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
