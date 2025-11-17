import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import asyncio

# --- State Definition ---
class AgentState(TypedDict):
    task_id: str
    product_idea: str
    status: str
    # This will hold the content for the human to approve
    pending_approval_content: Optional[str]

# --- In-Memory Task Storage ---
# In a real app, you'd use a database (e.g., Redis, PostgreSQL)
tasks = {}

# --- Agent Node Functions (Placeholders) ---
def intake_node(state: AgentState):
    """
    Represents the initial step where the product idea is processed.
    For now, it just moves the task to a pending approval state.
    """
    print(f"--- Node: intake_node (Task ID: {state['task_id']}) ---")
    state['status'] = "pending_approval"
    # This is the content the human will see
    state['pending_approval_content'] = f"Please approve the product idea: '{state['product_idea']}'"
    tasks[state['task_id']] = state # Update task state
    return state

def approved_node(state: AgentState):
    """Represents the state after human approval."""
    print(f"--- Node: approved_node (Task ID: {state['task_id']}) ---")
    state['status'] = "completed"
    tasks[state['task_id']] = state # Update task state
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
app_graph = workflow.compile()

# --- FastAPI Application ---
app = FastAPI(
    title="Multi-Agent Product Squad API",
    description="API for managing and interacting with the AI agent squad.",
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

class RespondToApprovalRequest(BaseModel):
    task_id: str
    approved: bool # Human's decision

# --- API Endpoints ---
@app.post("/start_task", response_model=TaskStatus)
async def start_task(request: StartTaskRequest):
    """
    Starts a new product validation task.
    """
    task_id = str(uuid.uuid4())
    initial_state = AgentState(
        task_id=task_id,
        product_idea=request.product_idea,
        status="starting",
        pending_approval_content=None
    )
    tasks[task_id] = initial_state

    # Asynchronously run the graph
    asyncio.create_task(app_graph.ainvoke(initial_state))

    return TaskStatus(**tasks[task_id])

@app.get("/get_pending_approval", response_model=Optional[TaskStatus])
def get_pending_approval():
    """
    Checks if there is a task waiting for human approval.
    """
    for task_id, state in tasks.items():
        if state['status'] == "pending_approval":
            return TaskStatus(**state)
    return None

@app.post("/respond_to_approval", response_model=TaskStatus)
def respond_to_approval(request: RespondToApprovalRequest):
    """
    Allows the human to approve or reject a task.
    """
    task = tasks.get(request.task_id)
    if not task or task['status'] != 'pending_approval':
        raise HTTPException(status_code=404, detail="Task not found or not pending approval.")

    if request.approved:
        # In a real graph, this would trigger the next step
        # For now, we manually update the state
        task['status'] = 'approved'
        # Here you would continue the graph execution
        print(f"Task {request.task_id} approved by human.")
        # For this simple version, we'll just move it to the final state
        asyncio.create_task(app_graph.ainvoke(task))
    else:
        task['status'] = 'rejected'
        print(f"Task {request.task_id} rejected by human.")

    tasks[request.task_id] = task
    return TaskStatus(**task)

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