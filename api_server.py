#!/usr/bin/env python3
"""
FastAPI Server for Gemini API Access
Provides REST endpoints for Gemini AI functionality
"""

import os
import asyncio
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from gemini_webapi import GeminiClient, set_log_level, logger
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import APIError, AuthError, UsageLimitExceeded, ModelInvalid, TemporarilyBlocked

# Configure logging
set_log_level("WARNING")  # Reduce log noise in production

# Global client instance (singleton pattern)
gemini_client: Optional[GeminiClient] = None
chat_sessions: Dict[str, Any] = {}  # Store chat sessions by ID

# Pydantic models for API requests/responses
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Message to send to Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature (if supported)")
    system_prompt: Optional[str] = Field(None, description="System prompt or gem ID")

class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: Optional[str] = None
    candidates_count: int
    thoughts: Optional[str] = None
    images: List[Dict[str, str]] = []
    metadata: Dict[str, Any] = {}

class ErrorResponse(BaseModel):
    error: str
    error_type: str
    details: Optional[Dict] = None

class ChatListResponse(BaseModel):
    sessions: List[Dict[str, Any]]
    total: int

class ModelInfo(BaseModel):
    name: str
    display_name: str
    advanced_only: bool
    available: bool

# Initialize FastAPI app
app = FastAPI(
    title="Gemini API Server",
    description="REST API for Google Gemini AI access",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_client() -> GeminiClient:
    """Get or initialize the Gemini client"""
    global gemini_client
    
    if gemini_client is None:
        secure_1psid = os.getenv("SECURE_1PSID")
        secure_1psidts = os.getenv("SECURE_1PSIDTS")
        
        if not secure_1psid:
            raise HTTPException(
                status_code=500, 
                detail="Gemini client not configured. Missing SECURE_1PSID environment variable."
            )
        
        try:
            gemini_client = GeminiClient(secure_1psid, secure_1psidts)
            await gemini_client.init(auto_refresh=True)
            logger.info("Gemini client initialized successfully")
        except AuthError as e:
            raise HTTPException(status_code=401, detail=f"Gemini authentication failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini client: {str(e)}")
    
    return gemini_client

def get_model_enum(model_name: str) -> Model:
    """Convert model name string to Model enum"""
    model_map = {
        "gemini-3.0-pro": Model.G_3_0_PRO,
        "gemini-2.5-pro": Model.G_2_5_PRO,
        "gemini-2.5-flash": Model.G_2_5_FLASH,
        "unspecified": Model.UNSPECIFIED,
    }
    return model_map.get(model_name, Model.G_2_5_FLASH)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "Gemini API Server", "version": "1.0.0"}

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        client = await get_client()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "client_initialized": client is not None,
            "active_sessions": len(chat_sessions)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available Gemini models"""
    models = []
    for model in Model:
        models.append(ModelInfo(
            name=model.model_name,
            display_name=model.model_name.replace("-", " ").title(),
            advanced_only=model.advanced_only,
            available=True  # We could check availability here
        ))
    return models

@app.post("/chat", response_model=ChatResponse)
async def send_message(request: MessageRequest):
    """Send a single message to Gemini"""
    try:
        client = await get_client()
        model = get_model_enum(request.model)
        
        # Generate response
        response = await client.generate_content(
            prompt=request.message,
            model=model
        )
        
        # Extract images
        images = []
        for img in response.images:
            images.append({
                "url": img.url,
                "title": img.title,
                "alt": img.alt,
                "type": "web" if hasattr(img, 'web_images') else "generated"
            })
        
        return ChatResponse(
            response=response.text,
            model=response.candidates[0].rcid,
            candidates_count=len(response.candidates),
            thoughts=response.thoughts,
            images=images,
            metadata={
                "rcid": response.rcid,
                "metadata": response.metadata
            }
        )
        
    except UsageLimitExceeded as e:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {str(e)}")
    except TemporarilyBlocked as e:
        raise HTTPException(status_code=403, detail=f"Access temporarily blocked: {str(e)}")
    except ModelInvalid as e:
        raise HTTPException(status_code=400, detail=f"Model not available: {str(e)}")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def send_chat_message(session_id: str, request: MessageRequest):
    """Send a message in a chat session"""
    try:
        client = await get_client()
        
        # Get or create chat session
        if session_id not in chat_sessions:
            # Create new session
            chat_sessions[session_id] = client.start_chat()
        
        chat = chat_sessions[session_id]
        model = get_model_enum(request.model)
        
        # Send message
        response = await chat.send_message(
            prompt=request.message,
            model=model
        )
        
        # Extract images
        images = []
        for img in response.images:
            images.append({
                "url": img.url,
                "title": img.title,
                "alt": img.alt,
                "type": "web" if hasattr(img, 'web_images') else "generated"
            })
        
        return ChatResponse(
            response=response.text,
            model=model.model_name,
            session_id=session_id,
            candidates_count=len(response.candidates),
            thoughts=response.thoughts,
            images=images,
            metadata={
                "rcid": response.rcid,
                "metadata": response.metadata,
                "chat_metadata": chat.metadata
            }
        )
        
    except UsageLimitExceeded as e:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {str(e)}")
    except TemporarilyBlocked as e:
        raise HTTPException(status_code=403, detail=f"Access temporarily blocked: {str(e)}")
    except ModelInvalid as e:
        raise HTTPException(status_code=400, detail=f"Model not available: {str(e)}")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/chat/new", response_model=Dict[str, str])
async def create_chat_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.get("/chat/{session_id}", response_model=Dict[str, Any])
async def get_chat_session(session_id: str):
    """Get chat session info"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chat = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "metadata": chat.metadata,
        "exists": True
    }

@app.delete("/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    del chat_sessions[session_id]
    return {"message": "Chat session deleted"}

@app.get("/chat", response_model=ChatListResponse)
async def list_chat_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, chat in chat_sessions.items():
        sessions.append({
            "session_id": session_id,
            "metadata": chat.metadata,
            "created_at": "unknown"  # Could add timestamp tracking
        })
    
    return ChatListResponse(sessions=sessions, total=len(sessions))

@app.post("/upload", response_model=Dict[str, Any])
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing with Gemini"""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            file_path = tmp_file.name
        
        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": len(content),
            "content_type": file.content_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/chat/{session_id}/files", response_model=ChatResponse)
async def send_message_with_files(
    session_id: str, 
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    model: str = Form(default="gemini-2.5-flash"),
    files: List[UploadFile] = File(...)
):
    """Send message with file attachments"""
    try:
        client = await get_client()
        
        # Get or create chat session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.start_chat()
        
        chat = chat_sessions[session_id]
        model = get_model_enum(model)
        
        # Save uploaded files temporarily
        file_paths = []
        temp_files = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file.flush()
                file_paths.append(tmp_file.name)
                temp_files.append(tmp_file.name)  # For cleanup
        
        try:
            # Generate response with files
            response = await chat.send_message(
                prompt=message,
                files=file_paths,
                model=model
            )
            
            # Extract images
            images = []
            for img in response.images:
                images.append({
                    "url": img.url,
                    "title": img.title,
                    "alt": img.alt,
                    "type": "web" if hasattr(img, 'web_images') else "generated"
                })
            
            return ChatResponse(
                response=response.text,
                model=model.model_name,
                session_id=session_id,
                candidates_count=len(response.candidates),
                thoughts=response.thoughts,
                images=images,
                metadata={
                    "rcid": response.rcid,
                    "metadata": response.metadata,
                    "files_processed": len(file_paths)
                }
            )
            
        finally:
            # Schedule cleanup of temp files
            def cleanup_files():
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            background_tasks.add_task(cleanup_files)
            
    except UsageLimitExceeded as e:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {str(e)}")
    except TemporarilyBlocked as e:
        raise HTTPException(status_code=403, detail=f"Access temporarily blocked: {str(e)}")
    except ModelInvalid as e:
        raise HTTPException(status_code=400, detail=f"Model not available: {str(e)}")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="http_error"
        ).dict()
    )

if __name__ == "__main__":
    # Check environment variables
    if not os.getenv("SECURE_1PSID"):
        print("ERROR: SECURE_1PSID environment variable is required!")
        print("Please set it before starting the server:")
        print("set SECURE_1PSID=your_cookie_value_here")
        print("set SECURE_1PSIDTS=your_cookie_value_here  # Optional")
        exit(1)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
