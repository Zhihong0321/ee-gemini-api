#!/usr/bin/env python3
"""
Production-ready Gemini API Server for Railway deployment
Optimized for cloud deployment with proper error handling and security
"""

import os
import json
import asyncio
import uuid
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import APIError, AuthError, UsageLimitExceeded, ModelInvalid, TemporarilyBlocked

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent

# Configure gemini logging to be less verbose
set_log_level("WARNING")

# Global client instance with lazy loading
gemini_client: Optional[GeminiClient] = None
chat_sessions: Dict[str, Any] = {}  

# Production environment variables
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
COOKIE_UPDATE_TOKEN = os.getenv("COOKIE_UPDATE_TOKEN")
COOKIES_FILE = Path(os.getenv("COOKIE_STORE_PATH", str(BASE_DIR / "data" / "cookies.json")))

# Pydantic models for production
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Message to send to Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt or gem ID")

class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: Optional[str] = None
    candidates_count: int
    thoughts: Optional[str] = None
    images: List[Dict[str, str]] = []
    metadata: Dict[str, Any] = {}
    success: bool = True

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_type: str
    timestamp: str = datetime.utcnow().isoformat()

class StatusResponse(BaseModel):
    status: str
    environment: str
    client_ready: bool
    active_sessions: int
    uptime: str
    version: str = "1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info(f"Starting Gemini API Server in {RAILWAY_ENVIRONMENT} mode")
    logger.info(f"Server will run on {HOST}:{PORT}")
    
    # Initialize client in background
    try:
        await get_or_init_client()
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gemini API Server")
    if gemini_client:
        await gemini_client.close()

def _load_cookies_from_store() -> Tuple[Optional[str], Optional[str]]:
    try:
        if COOKIES_FILE.exists():
            data = json.loads(COOKIES_FILE.read_text())
            return data.get("SECURE_1PSID"), data.get("SECURE_1PSIDTS")
    except Exception as exc:
        logger.warning(f"Failed to read cookie store: {exc}")
    return None, None


def _persist_cookies(secure_1psid: str, secure_1psidts: Optional[str]) -> None:
    try:
        COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        COOKIES_FILE.write_text(json.dumps({
            "SECURE_1PSID": secure_1psid,
            "SECURE_1PSIDTS": secure_1psidts
        }))
    except Exception as exc:
        logger.error(f"Failed to persist cookies: {exc}")
        raise HTTPException(status_code=500, detail="Unable to persist cookies")


def _get_stored_credentials() -> Tuple[Optional[str], Optional[str]]:
    env_psid = os.getenv("SECURE_1PSID")
    env_psidts = os.getenv("SECURE_1PSIDTS")
    if env_psid:
        return env_psid, env_psidts
    return _load_cookies_from_store()


async def get_or_init_client(force_reload: bool = False) -> GeminiClient:
    """Get or initialize the Gemini client with error handling"""
    global gemini_client
    
    if force_reload and gemini_client is not None:
        await gemini_client.close()
        gemini_client = None

    if gemini_client is None:
        secure_1psid, secure_1psidts = _get_stored_credentials()
        
        if not secure_1psid:
            raise ValueError("SECURE_1PSID environment variable is required")
        
        try:
            gemini_client = GeminiClient(secure_1psid, secure_1psidts)
            await gemini_client.init(auto_refresh=True)
            logger.info("Gemini client initialized successfully")
        except AuthError as e:
            logger.error(f"Gemini authentication failed: {e}")
            raise HTTPException(
                status_code=503, 
                detail="Service temporarily unavailable - authentication failed"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise HTTPException(
                status_code=503, 
                detail="Service temporarily unavailable"
            )
    
    return gemini_client

def get_model_enum(model_name: str) -> Model:
    """Convert model name string to Model enum with validation"""
    model_map = {
        "gemini-3.0-pro": Model.G_3_0_PRO,
        "gemini-2.5-pro": Model.G_2_5_PRO,
        "gemini-2.5-flash": Model.G_2_5_FLASH,
        "unspecified": Model.UNSPECIFIED,
    }
    return model_map.get(model_name, Model.G_2_5_FLASH)


def _extract_secure_cookies(cookie_payload: Any) -> Tuple[Optional[str], Optional[str]]:
    cookies = []
    if isinstance(cookie_payload, list):
        cookies = cookie_payload
    elif isinstance(cookie_payload, dict):
        if "cookies" in cookie_payload and isinstance(cookie_payload["cookies"], list):
            cookies = cookie_payload["cookies"]
        else:
            cookies = [cookie_payload]

    secure_1psid = None
    secure_1psidts = None

    for entry in cookies:
        name = entry.get("name") if isinstance(entry, dict) else None
        value = entry.get("value") if isinstance(entry, dict) else None
        if name == "__Secure-1PSID":
            secure_1psid = value
        elif name == "__Secure-1PSIDTS":
            secure_1psidts = value

    return secure_1psid, secure_1psidts

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Gemini API Server",
    description="Production REST API for Google Gemini AI access",
    version="1.0.0",
    lifespan=lifespan,
    # Uncomment to enable docs in production
    docs_url="/docs" if RAILWAY_ENVIRONMENT == "development" else "/docs",
    redoc_url="/redoc" if RAILWAY_ENVIRONMENT == "development" else "/redoc"
)

# Add production CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers for production"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    if RAILWAY_ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Gemini API Server",
        "version": "1.0.0",
        "environment": RAILWAY_ENVIRONMENT,
        "status": "running"
    }


@app.get("/cookies", response_class=HTMLResponse)
async def cookie_form():
    """Simple admin form to paste cookie JSON"""
    token_field = ""
    if COOKIE_UPDATE_TOKEN:
        token_field = """
            <label for=\"token\">Admin Token</label>
            <input type=\"password\" id=\"token\" name=\"token\" placeholder=\"Enter token\" required />
        """

    html_content = f"""
    <html>
        <head>
            <title>Update Gemini Cookies</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }}
                form {{ display: flex; flex-direction: column; gap: 1rem; }}
                textarea {{ width: 100%; height: 320px; font-family: monospace; }}
                .message {{ color: #666; font-size: 0.9rem; }}
                button {{ padding: 0.75rem; font-size: 1rem; cursor: pointer; }}
            </style>
        </head>
        <body>
            <h2>Paste Gemini Cookie JSON</h2>
            <p class=\"message\">Paste the export from your browser's cookie viewer. Only __Secure-1PSID and __Secure-1PSIDTS are stored.</p>
            <form method=\"post\" action=\"/cookies\">
                {token_field}
                <textarea name=\"cookie_json\" placeholder=\"[{{}}]\" required></textarea>
                <button type=\"submit\">Update Cookies</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/cookies")
async def update_cookies(cookie_json: str = Form(...), token: Optional[str] = Form(default=None)):
    """Accept cookie JSON, persist it, and reload the Gemini client."""
    if COOKIE_UPDATE_TOKEN and token != COOKIE_UPDATE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    try:
        payload = json.loads(cookie_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}")

    secure_1psid, secure_1psidts = _extract_secure_cookies(payload)

    if not secure_1psid:
        raise HTTPException(status_code=400, detail="__Secure-1PSID cookie not found")

    os.environ["SECURE_1PSID"] = secure_1psid
    if secure_1psidts:
        os.environ["SECURE_1PSIDTS"] = secure_1psidts

    _persist_cookies(secure_1psid, secure_1psidts)

    await get_or_init_client(force_reload=True)

    return {
        "success": True,
        "message": "Gemini cookies updated successfully",
        "updated_at": datetime.utcnow().isoformat(),
        "has_sidts": bool(secure_1psidts)
    }

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Production health check for Railway monitoring"""
    try:
        client_ready = gemini_client is not None
        return StatusResponse(
            status="healthy" if client_ready else "initializing",
            environment=RAILWAY_ENVIRONMENT,
            client_ready=client_ready,
            active_sessions=len(chat_sessions),
            uptime="running"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List available models with availability info"""
    try:
        models = []
        for model in Model:
            models.append({
                "name": model.model_name,
                "display_name": model.model_name.replace("-", " ").title(),
                "advanced_only": model.advanced_only,
                "available": True  # Could add availability check here
            })
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@app.post("/chat", response_model=ChatResponse)
async def send_message(request: MessageRequest):
    """Send a single message to Gemini with production error handling"""
    try:
        client = await get_or_init_client()
        model = get_model_enum(request.model)
        
        try:
            response = await client.generate_content(
                prompt=request.message,
                model=model
            )
        except UsageLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        except TemporarilyBlocked as e:
            logger.error(f"Access blocked: {e}")
            raise HTTPException(status_code=403, detail="Access temporarily blocked by Google")
        except ModelInvalid as e:
            logger.warning(f"Invalid model: {e}")
            raise HTTPException(status_code=400, detail=f"Model not available: {str(e)}")
        
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
            candidates_count=len(response.candidates),
            thoughts=response.thoughts,
            images=images,
            metadata={
                "rcid": response.rcid,
                "metadata_length": len(response.metadata)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in send_message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def send_chat_message(session_id: str, request: MessageRequest):
    """Send message in chat session with session management"""
    try:
        client = await get_or_init_client()
        
        # Get or create chat session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.start_chat()
        
        chat = chat_sessions[session_id]
        model = get_model_enum(request.model)
        
        try:
            response = await chat.send_message(
                prompt=request.message,
                model=model
            )
        except UsageLimitExceeded as e:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except TemporarilyBlocked as e:
            raise HTTPException(status_code=403, detail="Access temporarily blocked")
        
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
                "chat_metadata_length": len(chat.metadata)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat/new", response_model=Dict[str, str])
async def create_chat_session():
    """Create new chat session with unique ID"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = None  # Will be initialized on first message
    return {"session_id": session_id, "status": "created"}

@app.delete("/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete chat session and cleanup"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    del chat_sessions[session_id]
    return {"message": "Chat session deleted", "session_id": session_id}

@app.get("/status")
async def detailed_status():
    """Detailed status for monitoring"""
    try:
        client_stats = {
            "initialized": gemini_client is not None,
            "active_sessions": len(chat_sessions),
            "environment": RAILWAY_ENVIRONMENT,
            "server_uptime": "running",
        }
        
        # Test client functionality if available
        if gemini_client:
            try:
                # Quick test call
                test_response = await gemini_client.generate_content(
                    "Just say 'OK' - this is a health check.",
                    model=Model.G_2_5_FLASH
                )
                client_stats["api_test"] = "passed"
                client_stats["last_test"] = datetime.utcnow().isoformat()
            except Exception as e:
                client_stats["api_test"] = "failed"
                client_stats["error"] = str(e)
        
        return client_stats
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"error": str(e)}

# Production exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="http_error"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type="unexpected_error"
        ).dict()
    )

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "production_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
        reload=RAILWAY_ENVIRONMENT == "development"
    )
