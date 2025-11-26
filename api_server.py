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
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!doctype html>
    <html lang=\"en\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Gemini API Server</title>
        <script src=\"https://cdn.tailwindcss.com\"></script>
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
        <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap\" rel=\"stylesheet\" />
        <style> body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }} </style>
      </head>
      <body class=\"bg-gray-50\">
        <div class=\"min-h-screen\">
          <header class=\"bg-white border-b\">
            <div class=\"max-w-5xl mx-auto px-4 py-4 flex items-center justify-between\">
              <div class=\"flex items-center gap-2\">
                <div class=\"h-8 w-8 rounded-lg bg-indigo-600\"></div>
                <div>
                  <h1 class=\"text-lg font-semibold\">Gemini API Server</h1>
                  <p class=\"text-xs text-gray-500\">Version: 1.0.0</p>
                </div>
              </div>
              <div class=\"flex items-center gap-2\">
                <a href=\"/docs\" class=\"inline-flex items-center rounded-md bg-gray-900 text-white text-sm px-3 py-2 hover:bg-black\">API Docs</a>
                <a href=\"/redoc\" class=\"inline-flex items-center rounded-md bg-white border text-sm px-3 py-2 hover:bg-gray-100\">ReDoc</a>
              </div>
            </div>
          </header>
          <main class=\"max-w-5xl mx-auto px-4 py-8\">
            <div class=\"grid grid-cols-1 lg:grid-cols-3 gap-6\">
              <section class=\"lg:col-span-2\">
                <div class=\"rounded-xl border bg-white p-6 shadow-sm\">
                  <div class=\"flex items-center justify-between\">
                    <h2 class=\"text-base font-semibold\">Server Health</h2>
                    <button id=\"refreshBtn\" class=\"text-sm px-3 py-1.5 rounded-md border hover:bg-gray-50\">Refresh</button>
                  </div>
                  <div class=\"mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4\">
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Status</p>
                      <p id=\"hStatus\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Client Initialized</p>
                      <p id=\"hClient\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Active Sessions</p>
                      <p id=\"hSessions\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                  </div>
                  <div class=\"mt-6 rounded-lg border p-4\">
                    <p class=\"text-sm font-medium\">Detailed Health</p>
                    <pre id=\"statusJson\" class=\"mt-2 bg-gray-50 rounded p-3 text-xs overflow-x-auto\"></pre>
                  </div>
                </div>
              </section>
              <section>
                <div class=\"rounded-xl border bg-white p-6 shadow-sm\">
                  <h2 class=\"text-base font-semibold\">Paste Cookie JSON</h2>
                  <p class=\"mt-1 text-xs text-gray-500\">Only __Secure-1PSID and __Secure-1PSIDTS are stored.</p>
                  <form id=\"cookieForm\" class=\"mt-4 space-y-3\">
                    <label class=\"block text-sm font-medium text-gray-700\">Cookie JSON</label>
                    <textarea name=\"cookie_json\" id=\"cookie_json\" class=\"mt-1 w-full h-40 rounded-md border border-gray-300 px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500\" placeholder=\"[{{}}]\" required></textarea>
                    <button type=\"submit\" class=\"w-full inline-flex items-center justify-center rounded-md bg-indigo-600 text-white text-sm px-3 py-2 hover:bg-indigo-700\">Update Cookies</button>
                  </form>
                  <div id=\"cookieMsg\" class=\"mt-3 text-sm\"></div>
                </div>
              </section>
            </div>
          </main>
          <footer class=\"mt-8 py-6 text-center text-xs text-gray-500\">Gemini API Server</footer>
        </div>
        <script>
        async function loadHealth() {
          try {
            const h = await fetch('/health').then(r => r.json());
            document.getElementById('hStatus').textContent = h.status;
            document.getElementById('hClient').textContent = String(h.client_initialized);
            document.getElementById('hSessions').textContent = String(h.active_sessions);
          } catch(e) {}
          try {
            const s = await fetch('/health').then(r => r.json());
            document.getElementById('statusJson').textContent = JSON.stringify(s, null, 2);
          } catch(e) {}
        }
        document.getElementById('refreshBtn').addEventListener('click', loadHealth);
        loadHealth();
        document.getElementById('cookieForm').addEventListener('submit', async (ev) => {
          ev.preventDefault();
          const el = document.getElementById('cookieMsg');
          el.textContent = '';
          const fd = new FormData(ev.target);
          try {
            const res = await fetch('/cookies', { method: 'POST', body: fd });
            const data = await res.json();
            if (data && data.success) {
              el.textContent = 'Cookies updated successfully';
              el.className = 'mt-3 text-sm text-green-600';
              loadHealth();
            } else {
              el.textContent = (data && data.error) ? data.error : 'Update failed';
              el.className = 'mt-3 text-sm text-red-600';
            }
          } catch(e) {
            el.textContent = String(e);
            el.className = 'mt-3 text-sm text-red-600';
          }
        });
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
@app.post("/cookies")
async def update_cookies(cookie_json: str = Form(...)):
    try:
        import json as _json
        payload = _json.loads(cookie_json)
        cookies = []
        if isinstance(payload, list):
            cookies = payload
        elif isinstance(payload, dict):
            cookies = payload.get("cookies", []) if isinstance(payload.get("cookies"), list) else [payload]
        sid = None
        sidts = None
        for c in cookies:
            name = c.get("name") if isinstance(c, dict) else None
            val = c.get("value") if isinstance(c, dict) else None
            if name == "__Secure-1PSID":
                sid = val
            elif name == "__Secure-1PSIDTS":
                sidts = val
        if not sid:
            raise HTTPException(status_code=400, detail="__Secure-1PSID cookie not found")
        os.environ["SECURE_1PSID"] = sid
        if sidts:
            os.environ["SECURE_1PSIDTS"] = sidts
        global gemini_client
        gemini_client = None
        await get_client()
        return {"success": True, "has_sidts": bool(sidts)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
