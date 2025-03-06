"""
API Interface for Legion AGI

This module provides a RESTful API for interacting with the Legion AGI system.
It supports both synchronous and asynchronous query processing, session management,
agent inspection, and visualization access.
"""

import os
import sys
import time
import uuid
import json
import base64
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

import fastapi
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from loguru import logger

from legion_agi.main import LegionAGI
from legion_agi.config import (
    DEFAULT_MODEL,
    TOOL_MODE,
    EVOLUTION_MODE,
    DATA_DIR
)
from legion_agi.utils.logger import LogContext, PerformanceTimer


# API models for request/response
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., title="Query text", min_length=1)
    method: str = Field("past", title="Reasoning method", 
                       description="Reasoning method to use: past, raft, or eat")
    parameters: Optional[Dict[str, Any]] = Field(None, title="Additional parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Explain the relationship between quantum mechanics and consciousness",
                "method": "past",
                "parameters": {
                    "temperature": 0.7
                }
            }
        }


class SessionRequest(BaseModel):
    """Session creation/management request model."""
    model: str = Field(DEFAULT_MODEL, title="LLM model to use")
    mode: str = Field(TOOL_MODE, title="Operation mode (tool or evolve)")
    visualize: bool = Field(True, title="Enable visualizations")
    load_state: Optional[str] = Field(None, title="Path to state file to load")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "llama3.1:8b",
                "mode": "tool",
                "visualize": True,
                "load_state": None
            }
        }


class AgentRequest(BaseModel):
    """Agent creation/management request model."""
    agent_type: str = Field(..., title="Type of agent to create")
    customizations: Optional[Dict[str, Any]] = Field(None, title="Agent customizations")
    
    class Config:
        schema_extra = {
            "example": {
                "agent_type": "science",
                "customizations": {
                    "name": "Dr. Marie Johnson",
                    "role": "Physicist",
                    "style": "Analytical and precise"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., title="Error message")
    detail: Optional[str] = Field(None, title="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Session not found",
                "detail": "The requested session does not exist or has expired"
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="Legion AGI API",
    description="API for interacting with the Legion AGI system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, LegionAGI] = {}
session_locks: Dict[str, threading.Lock] = {}
async_tasks: Dict[str, asyncio.Task] = {}
session_results: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> LegionAGI:
    """
    Get session by ID or raise 404 exception.
    
    Args:
        session_id: Session ID
        
    Returns:
        Legion AGI instance
        
    Raises:
        HTTPException: If session not found
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


async def async_process_query(
    session_id: str,
    query: str,
    method: str,
    parameters: Optional[Dict[str, Any]] = None
) -> None:
    """
    Process query asynchronously.
    
    Args:
        session_id: Session ID
        query: Query text
        method: Reasoning method
        parameters: Additional parameters
    """
    session = sessions.get(session_id)
    if not session:
        session_results[session_id] = {
            "error": "Session not found or expired",
            "status": "error"
        }
        return
        
    # Default parameters
    params = parameters or {}
    
    try:
        # Get session lock
        lock = session_locks.get(session_id)
        if lock and not lock.acquire(blocking=False):
            session_results[session_id] = {
                "error": "Session is busy processing another query",
                "status": "error"
            }
            return
            
        try:
            # Update status
            session_results[session_id] = {
                "status": "processing",
                "progress": "starting",
                "timestamp": datetime.now().isoformat()
            }
            
            # Process query with periodic progress updates
            result = None
            
            # Run in thread to avoid blocking the event loop
            def run_query():
                nonlocal result
                try:
                    # Process with selected method
                    result = session.process_query(query, method)
                except Exception as e:
                    logger.exception(f"Error processing query: {e}")
                    result = {"error": str(e), "status": "error"}
                    
            # Start processing thread
            thread = threading.Thread(target=run_query)
            thread.start()
            
            # Update progress while waiting
            progress_states = [
                "analyzing query",
                "spawning agents",
                "collaborative reasoning",
                "refining solutions",
                "integrating results",
                "finalizing response"
            ]
            
            i = 0
            while thread.is_alive():
                # Update progress
                session_results[session_id] = {
                    "status": "processing",
                    "progress": progress_states[i % len(progress_states)],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Wait a bit
                i += 1
                await asyncio.sleep(1.0)
                
            # Wait for thread to complete
            thread.join()
            
            # Store final result
            if result:
                session_results[session_id] = {
                    "status": "completed",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                session_results[session_id] = {
                    "status": "error",
                    "error": "No result returned",
                    "timestamp": datetime.now().isoformat()
                }
                
        finally:
            # Release lock
            if lock:
                lock.release()
                
    except Exception as e:
        logger.exception(f"Error in async query processing: {e}")
        session_results[session_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/", tags=["General"])
async def get_api_info():
    """Get API information."""
    return {
        "name": "Legion AGI API",
        "version": "1.0.0",
        "description": "API for interacting with the Legion AGI system"
    }


@app.post("/sessions", tags=["Sessions"], response_model=Dict[str, Any])
async def create_session(request: SessionRequest):
    """
    Create a new Legion AGI session.
    
    Args:
        request: Session configuration
        
    Returns:
        Session information including ID
    """
    try:
        with LogContext("create_session", **request.dict()):
            with PerformanceTimer("create_session", threshold_ms=500):
                # Generate session ID
                session_id = str(uuid.uuid4())
                
                # Create data directory for this session
                session_dir = os.path.join(DATA_DIR, f"session_{session_id}")
                os.makedirs(session_dir, exist_ok=True)
                
                # Initialize Legion AGI
                legion = LegionAGI(
                    model=request.model,
                    mode=request.mode,
                    session_id=session_id,
                    data_dir=DATA_DIR,
                    visualize=request.visualize
                )
                
                # Load state if specified
                if request.load_state and os.path.exists(request.load_state):
                    legion.load_state(request.load_state)
                    
                # Start Legion AGI
                legion.running = True
                
                # Store session
                sessions[session_id] = legion
                session_locks[session_id] = threading.Lock()
                session_results[session_id] = {
                    "status": "ready",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Start evolution thread if in evolution mode
                if request.mode == EVOLUTION_MODE:
                    def run_evolution():
                        try:
                            legion._run_evolution_mode()
                        except Exception as e:
                            logger.exception(f"Error in evolution thread: {e}")
                            
                    # Start evolution thread
                    thread = threading.Thread(target=run_evolution, daemon=True)
                    thread.start()
                    
                logger.info(f"Created session: {session_id}")
                
                # Return session information
                return {
                    "session_id": session_id,
                    "status": "ready",
                    "config": request.dict(),
                    "created_at": datetime.now().isoformat()
                }
                
    except Exception as e:
        logger.exception(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", tags=["Sessions"], response_model=Dict[str, Any])
async def get_session_info(session_id: str = Path(..., description="Session ID")):
    """
    Get information about a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        Session information
    """
    try:
        session = get_session(session_id)
        
        # Get system status
        status = session.get_system_status()
        
        # Include processing status if available
        result_status = session_results.get(session_id, {}).get("status", "unknown")
        
        return {
            "session_id": session_id,
            "status": result_status,
            "system_status": status,
            "updated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str = Path(..., description="Session ID")):
    """
    Delete a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        Success message
    """
    try:
        # Check if session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Stop Legion AGI
        session = sessions[session_id]
        if session.running:
            session.stop()
            
        # Clean up tasks
        if session_id in async_tasks and not async_tasks[session_id].done():
            async_tasks[session_id].cancel()
            
        # Remove session
        del sessions[session_id]
        if session_id in session_locks:
            del session_locks[session_id]
        if session_id in async_tasks:
            del async_tasks[session_id]
        if session_id in session_results:
            del session_results[session_id]
            
        logger.info(f"Deleted session: {session_id}")
        
        return {"status": "success", "message": "Session deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/query", tags=["Queries"], response_model=Dict[str, Any])
async def process_query_sync(
    request: QueryRequest,
    session_id: str = Path(..., description="Session ID")
):
    """
    Process a query synchronously.
    
    Args:
        request: Query request
        session_id: Session ID
        
    Returns:
        Query result
    """
    try:
        with LogContext("process_query", session_id=session_id, method=request.method):
            with PerformanceTimer("process_query", threshold_ms=500):
                session = get_session(session_id)
                
                # Get session lock
                lock = session_locks.get(session_id)
                if lock and not lock.acquire(blocking=False):
                    raise HTTPException(
                        status_code=409,
                        detail="Session is busy processing another query"
                    )
                    
                try:
                    # Process query
                    result = session.process_query(request.query, request.method)
                    
                    # Store result
                    session_results[session_id] = {
                        "status": "completed",
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return {
                        "status": "completed",
                        "result": result,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                finally:
                    # Release lock
                    if lock:
                        lock.release()
                        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/query/async", 
    tags=["Queries"], 
    response_model=Dict[str, Any]
)
async def process_query_async(
    background_tasks: BackgroundTasks,
    request: QueryRequest,
    session_id: str = Path(..., description="Session ID")
):
    """
    Process a query asynchronously.
    
    Args:
        background_tasks: FastAPI background tasks
        request: Query request
        session_id: Session ID
        
    Returns:
        Task information
    """
    try:
        # Check if session exists
        get_session(session_id)
        
        # Check if there's already a task running
        if session_id in async_tasks and not async_tasks[session_id].done():
            return {
                "status": "processing",
                "message": "Already processing a query for this session",
                "task_id": str(id(async_tasks[session_id])),
                "session_id": session_id
            }
            
        # Start async task
        task = asyncio.create_task(
            async_process_query(
                session_id,
                request.query,
                request.method,
                request.parameters
            )
        )
        async_tasks[session_id] = task
        
        # Initialize result
        session_results[session_id] = {
            "status": "processing",
            "progress": "starting",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "processing",
            "message": "Query processing started",
            "task_id": str(id(task)),
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error starting async query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/query/status", 
    tags=["Queries"], 
    response_model=Dict[str, Any]
)
async def get_query_status(session_id: str = Path(..., description="Session ID")):
    """
    Get the status of the latest query.
    
    Args:
        session_id: Session ID
        
    Returns:
        Query status
    """
    try:
        # Check if session exists
        get_session(session_id)
        
        # Get query status
        status = session_results.get(session_id, {
            "status": "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "session_id": session_id,
            "status": status.get("status", "unknown"),
            "progress": status.get("progress", ""),
            "timestamp": status.get("timestamp", datetime.now().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting query status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/query/result", 
    tags=["Queries"], 
    response_model=Dict[str, Any]
)
async def get_query_result(session_id: str = Path(..., description="Session ID")):
    """
    Get the result of the latest query.
    
    Args:
        session_id: Session ID
        
    Returns:
        Query result
    """
    try:
        # Check if session exists
        get_session(session_id)
        
        # Get query result
        status = session_results.get(session_id, {
            "status": "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
        if status.get("status") != "completed":
            return {
                "session_id": session_id,
                "status": status.get("status", "unknown"),
                "progress": status.get("progress", ""),
                "timestamp": status.get("timestamp", datetime.now().isoformat())
            }
            
        return {
            "session_id": session_id,
            "status": "completed",
            "result": status.get("result", {}),
            "timestamp": status.get("timestamp", datetime.now().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting query result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/agents", 
    tags=["Agents"], 
    response_model=List[Dict[str, Any]]
)
async def get_session_agents(session_id: str = Path(..., description="Session ID")):
    """
    Get all agents in a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        List of agents
    """
    try:
        session = get_session(session_id)
        
        if not hasattr(session, 'agents') or not session.agents:
            return []
            
        # Convert agents to JSON-serializable format
        agents = []
        for agent in session.agents:
            agents.append({
                "id": agent.agent_id,
                "name": agent.name,
                "role": agent.role,
                "cognitive_parameters": {
                    "creativity": agent.creativity,
                    "attention_span": agent.attention_span,
                    "learning_rate": agent.learning_rate
                }
            })
            
        return agents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting session agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/agents", 
    tags=["Agents"], 
    response_model=Dict[str, Any]
)
async def spawn_agent(
    request: AgentRequest,
    session_id: str = Path(..., description="Session ID")
):
    """
    Spawn a new agent in a session.
    
    Args:
        request: Agent request
        session_id: Session ID
        
    Returns:
        New agent information
    """
    try:
        session = get_session(session_id)
        
        if not hasattr(session, 'agent_spawner'):
            raise HTTPException(
                status_code=400,
                detail="Session does not support agent spawning"
            )
            
        # Spawn agent
        agent = session.agent_spawner.spawn_agent_by_type(
            request.agent_type,
            request.customizations
        )
        
        if not agent:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to spawn agent of type: {request.agent_type}"
            )
            
        return {
            "id": agent.agent_id,
            "name": agent.name,
            "role": agent.role,
            "cognitive_parameters": {
                "creativity": agent.creativity,
                "attention_span": agent.attention_span,
                "learning_rate": agent.learning_rate
            },
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error spawning agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/agents/{agent_id}", 
    tags=["Agents"], 
    response_model=Dict[str, Any]
)
async def get_agent_details(
    session_id: str = Path(..., description="Session ID"),
    agent_id: str = Path(..., description="Agent ID")
):
    """
    Get details of a specific agent.
    
    Args:
        session_id: Session ID
        agent_id: Agent ID
        
    Returns:
        Agent details
    """
    try:
        session = get_session(session_id)
        
        if not hasattr(session, 'agents') or not session.agents:
            raise HTTPException(status_code=404, detail="No agents found in session")
            
        # Find agent
        agent = None
        for a in session.agents:
            if a.agent_id == agent_id:
                agent = a
                break
                
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
        # Get agent details
        return {
            "id": agent.agent_id,
            "name": agent.name,
            "role": agent.role,
            "backstory": agent.backstory,
            "style": agent.style,
            "instructions": agent.instructions,
            "model": agent.model,
            "cognitive_parameters": {
                "creativity": agent.creativity,
                "attention_span": agent.attention_span,
                "learning_rate": agent.learning_rate
            },
            "memory_stats": {
                "stm_size": len(agent.memory_module.short_term_memory),
                "ltm_size": len(agent.memory_module.long_term_memory),
                "episodic_size": len(agent.memory_module.episodic_memory),
                "semantic_size": len(agent.memory_module.semantic_memory)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting agent details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/save", 
    tags=["Sessions"], 
    response_model=Dict[str, Any]
)
async def save_session_state(
    session_id: str = Path(..., description="Session ID"),
    filename: Optional[str] = Query(None, description="Custom filename")
):
    """
    Save the current state of a session.
    
    Args:
        session_id: Session ID
        filename: Optional custom filename
        
    Returns:
        Save result
    """
    try:
        session = get_session(session_id)
        
        # Save state
        state_file = session.save_state()
        
        return {
            "status": "success",
            "message": "Session state saved",
            "file_path": state_file,
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error saving session state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/visualizations", 
    tags=["Visualizations"], 
    response_model=List[Dict[str, Any]]
)
async def get_visualizations(
    session_id: str = Path(..., description="Session ID"),
    type: Optional[str] = Query(None, description="Visualization type")
):
    """
    Get available visualizations for a session.
    
    Args:
        session_id: Session ID
        type: Optional visualization type filter
        
    Returns:
        List of available visualizations
    """
    try:
        session = get_session(session_id)
        
        if not session.visualize:
            return []
            
        # Get visualization directory
        viz_dir = os.path.join(session.session_dir, "viz")
        if not os.path.exists(viz_dir):
            return []
            
        # Get all visualization files
        files = os.listdir(viz_dir)
        
        # Filter based on type
        if type:
            files = [f for f in files if type.lower() in f.lower()]
            
        # Convert to response format
        visualizations = []
        for file in files:
            file_path = os.path.join(viz_dir, file)
            size = os.path.getsize(file_path)
            created = datetime.fromtimestamp(os.path.getctime(file_path))
            
            visualizations.append({
                "filename": file,
                "path": file_path,
                "size": size,
                "created": created.isoformat(),
                "url": f"/sessions/{session_id}/visualizations/{file}"
            })
            
        return visualizations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/visualizations/{filename}", 
    tags=["Visualizations"]
)
async def get_visualization(
    session_id: str = Path(..., description="Session ID"),
    filename: str = Path(..., description="Visualization filename")
):
    """
    Get a specific visualization file.
    
    Args:
        session_id: Session ID
        filename: Visualization filename
        
    Returns:
        Visualization file
    """
    try:
        session = get_session(session_id)
        
        # Construct file path
        file_path = os.path.join(session.session_dir, "viz", filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Visualization not found")
            
        # Return file
        return FileResponse(file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server.
    
    Args:
        host: Host address
        port: Port number
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Legion AGI API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    # Start API server
    start_api_server(args.host, args.port)
