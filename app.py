import os
import uuid
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.graph import StateGraph, END 
from typing import TypedDict, List, Optional
import asyncio
import logging
from sqlalchemy import create_engine, Column, String, Text, LargeBinary
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SQLAlchemy Logging ---
# This will log all SQL statements issued by SQLAlchemy to the console.
# Set to logging.DEBUG to see result rows as well.
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# --- Database Setup (SQLite) ---
DATABASE_URL = "sqlite:///./tasks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DB Model for Tasks ---
class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(String, primary_key=True, index=True)
    product_idea = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    pending_approval_content = Column(Text, nullable=True)
    checkpoint = Column(LargeBinary, nullable=True) # To store graph state

# Create the database tables
Base.metadata.create_all(bind=engine)

# --- DB Dependency for FastAPI ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- State Definition ---
class AgentState(TypedDict):
    task_id: str
    product_idea: str
    status: str
    pending_approval_content: Optional[str]
    checkpoint: Optional[bytes] = None # Add checkpoint to state

# --- Agent Node Functions (Placeholders) ---
def intake_node(state: AgentState):
    """
    Represents the initial step where the product idea is processed.
    For now, it just moves the task to a pending approval state.
    """
    logger.info(f"--- Node: intake_node (Task ID: {state['task_id']}) ---")
    state['status'] = "pending_approval"
    # This is the content the human will see
    state['pending_approval_content'] = f"Please approve the product idea: '{state['product_idea']}'"
    return state

def approved_node(state: AgentState):
    """Represents the state after human approval."""
    logger.info(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "completed"
    return state

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("intake", intake_node)
workflow.add_node("approved", approved_node)

# --- Edge Definition ---
workflow.set_entry_point("intake")
workflow.add_edge("intake", "approved") # Simple linear flow for now
workflow.add_edge("approved", END)

# Compile the graph
# This is the key change: we tell the graph to interrupt execution
# *before* the "approved" node is run. This forces it to wait for
# human input.
app_graph = workflow.compile(interrupt_before=["approved"])

# --- FastAPI Application ---
app = FastAPI(
    title="Multi-Agent Product Squad API",
    description="API for managing and interacting with the AI agent squad.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allows the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity
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

    # Pydantic V2 config to allow creating model from ORM objects
    class Config:
        from_attributes = True

class RespondToApprovalRequest(BaseModel):
    task_id: str
    approved: bool # Human's decision

# --- API Endpoints ---
@app.post("/start_task", response_model=TaskStatus)
async def start_task(request: StartTaskRequest, db: Session = Depends(get_db)):
    """
    Starts a new product validation task.
    """
    logger.info(f"Received request to start task for idea: '{request.product_idea[:50]}...'")
    task_id = str(uuid.uuid4())
    initial_state = AgentState(
        task_id=task_id,
        product_idea=request.product_idea,
        status="starting",
        pending_approval_content=None
    )

    # Create and save the initial task record in the DB
    db_task = Task(**initial_state)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    logger.info(f"Task {task_id} created and saved to DB with status 'starting'.")

    # Asynchronously run the graph until the first interruption
    async def run_graph():
        logger.info(f"Graph for task {task_id} starting execution.")
        # This background task needs its own database session
        with SessionLocal() as db_session:
            # Use astream_events to get the checkpoint when the graph is interrupted
            last_state = None
            async for event in app_graph.astream_events(initial_state, version="v1"):
                kind = event["event"]
                if kind == "on_chain_stream":
                    # Capture the output of the last node that ran before interruption
                    if event["name"] == "intake":
                        last_state = event["data"]["chunk"]
                elif kind == "on_chain_end" and event["metadata"].get("interrupted"):
                    # Graph was interrupted, now we can save the state and checkpoint
                    task_to_update = db_session.query(Task).filter(Task.task_id == task_id).first()
                    if task_to_update and last_state:
                        task_to_update.status = last_state['status']
                        task_to_update.pending_approval_content = last_state['pending_approval_content']
                        task_to_update.checkpoint = event["metadata"]["checkpoint"]
                        db_session.commit()
                        logger.info(f"Graph for task {task_id} paused. DB updated with status '{task_to_update.status}' and checkpoint.")

    asyncio.create_task(run_graph())

    return TaskStatus(**initial_state)

@app.get("/get_pending_approval", response_model=Optional[TaskStatus])
def get_pending_approval(db: Session = Depends(get_db)):
    """
    Checks if there is a task waiting for human approval.
    """
    pending_task = db.query(Task).filter(Task.status == "pending_approval").first()
    if pending_task:
        logger.debug(f"Found pending task: {pending_task.task_id}")
        return TaskStatus.from_orm(pending_task)
    logger.debug("No pending tasks found.")
    return None

@app.post("/respond_to_approval", response_model=TaskStatus)
async def respond_to_approval(request: RespondToApprovalRequest, db: Session = Depends(get_db)):
    """
    Allows the human to approve or reject a task.
    """
    db_task = db.query(Task).filter(Task.task_id == request.task_id).first()
    if not db_task or db_task.status != 'pending_approval':
        logger.warning(f"Approval response for invalid or non-pending task {request.task_id} received.")
        raise HTTPException(status_code=404, detail="Task not found or not pending approval.")
    
    if request.approved:
        logger.info(f"Task {request.task_id} approved by human. Resuming graph.")
        # Load the checkpoint from the database to resume the graph
        checkpoint = db_task.checkpoint
        
        # Resume the graph from the checkpoint
        final_state = await app_graph.ainvoke(None, {"checkpoint": checkpoint})

        # Update the DB with the final state from the graph
        db_task.status = final_state.get('status', 'completed') # Use .get for safety
        db_task.pending_approval_content = None # Clear the approval content
        db_task.checkpoint = None # Clear the used checkpoint
        db.commit()
        logger.info(f"Graph for task {request.task_id} finished. Final status: '{db_task.status}'.")
        return TaskStatus.from_orm(db_task)
    else:
        logger.info(f"Task {request.task_id} rejected by human.")
        db_task.status = 'rejected'
        db_task.pending_approval_content = None # Clear the approval content
        db_task.checkpoint = None # Clear the unused checkpoint
        db.commit()
        # If rejected, we don't continue the graph.
        return TaskStatus.from_orm(db_task)

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
async def read_index():
    """Serves the main HTML page."""
    return FileResponse("index.html")

@app.get("/status")
def read_root():
    return {"message": "Welcome to the Multi-Agent Product Squad API!"}

# To run this app:
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload