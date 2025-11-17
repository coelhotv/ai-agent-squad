# --- Core Imports ---
import uuid
import asyncio
import logging

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Pydantic Imports ---
from pydantic import BaseModel

# --- LangGraph Imports ---
# Note: You will need to add 'langgraph[sqlite]' to your requirements.txt
# pip install "langgraph[sqlite]"
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Typing Imports ---
from typing import TypedDict, Optional

# --- Database Imports ---
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# --- Application Database (SQLite) ---
# This DB stores the high-level status of tasks for the UI
DATABASE_URL = "sqlite:///./tasks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(String, primary_key=True, index=True)
    product_idea = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    pending_approval_content = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

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

# The checkpointer is responsible for saving and loading the state of the graph
# It uses a separate SQLite database to store the checkpoints
memory_saver = SqliteSaver.from_conn_string("sqlite:///./checkpoints.sqlite")

# --- Agent Node Functions ---
# These nodes now only focus on modifying the state dictionary.
# The main application logic will handle database updates.
def intake_node(state: AgentState):
    logger.info(f"--- Node: intake_node (Task ID: {state['task_id']}) ---")
    state['status'] = "pending_approval"
    state['pending_approval_content'] = f"Please approve the product idea: '{state['product_idea']}'"
    return state

def approved_node(state: AgentState):
    logger.info(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "completed"
    return state

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("intake", intake_node)
workflow.add_node("approved", approved_node)
workflow.set_entry_point("intake")
workflow.add_edge("intake", "approved")
workflow.add_edge("approved", END)

# Compile the graph with the checkpointer and interruption
app_graph = workflow.compile(
    checkpointer=memory_saver,
    interrupt_before=["approved"]
)

# --- FastAPI Application ---
app = FastAPI(title="Multi-Agent Product Squad API", version="1.0.0")
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
    class Config:
        from_attributes = True

class RespondToApprovalRequest(BaseModel):
    task_id: str
    approved: bool

# --- API Endpoints (Refactored) ---
@app.post("/start_task", response_model=TaskStatus)
async def start_task(request: StartTaskRequest, db: Session = Depends(get_db)):
    logger.info(f"Received request to start task for idea: '{request.product_idea[:50]}...'")
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
        pending_approval_content=None
    )
    config = {"configurable": {"thread_id": task_id}}

    # 3. Asynchronously invoke the graph. It will run until it hits the interruption.
    # The checkpointer automatically saves its state.
    await app_graph.ainvoke(initial_state, config)
    logger.info(f"Graph for task {task_id} executed until interruption.")

    # 4. Get the state of the graph at the interruption point
    interrupted_state = app_graph.get_state(config)
    logger.info(f"Current graph state for task {task_id}: {interrupted_state.values}")

    # 5. Update our application DB with the new status from the graph
    db_task.status = interrupted_state.values['status']
    db_task.pending_approval_content = interrupted_state.values['pending_approval_content']
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
    
    # 2. Invoke the graph again. The checkpointer loads the state automatically.
    # We pass `None` as the state because the checkpointer is handling it.
    final_state = await app_graph.ainvoke(None, config)
    logger.info(f"Graph for task {request.task_id} finished execution.")

    # 3. Update our application DB with the final status
    db_task.status = final_state.get('status', 'completed')
    db_task.pending_approval_content = None
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
