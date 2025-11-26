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
GEMS_FILE = Path(os.getenv("GEMS_STORE_PATH", str(BASE_DIR / "data" / "gems.json")))

# Pydantic models for production
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Message to send to Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt or full gem:// URL (e.g. from gemini.google.com/gem/ID share link). Create/edit Gems on web.")

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

def _load_gems_from_store() -> List[Dict[str, Any]]:
    try:
        if GEMS_FILE.exists():
            return json.loads(GEMS_FILE.read_text())
    except Exception:
        return []
    return []

def _persist_gems(gems: List[Dict[str, Any]]) -> None:
    try:
        GEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        GEMS_FILE.write_text(json.dumps(gems))
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to persist gems")


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

@app.get("/", response_class=HTMLResponse)
async def root():
    token_field = ""
    if COOKIE_UPDATE_TOKEN:
        token_field = """
            <label class=\"block text-sm font-medium text-gray-700\">Admin Token</label>
            <input type=\"password\" id=\"token\" name=\"token\" placeholder=\"Enter token\" class=\"mt-1 w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500\" required />
        """

    html_content = f"""
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
        <style>
          body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
        </style>
      </head>
      <body class=\"bg-gray-50\">
        <div class=\"min-h-screen\">
          <header class=\"bg-white border-b\">
            <div class=\"max-w-5xl mx-auto px-4 py-4 flex items-center justify-between\">
              <div class=\"flex items-center gap-2\">
                <div class=\"h-8 w-8 rounded-lg bg-indigo-600\"></div>
                <div>
                  <h1 class=\"text-lg font-semibold\">Gemini API Server</h1>
                  <p class=\"text-xs text-gray-500\">Environment: {RAILWAY_ENVIRONMENT}</p>
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
                  <div id=\"health\" class=\"mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4\">
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Status</p>
                      <p id=\"hStatus\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Client Ready</p>
                      <p id=\"hClient\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Active Sessions</p>
                      <p id=\"hSessions\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                    <div class=\"rounded-lg border p-4\">
                      <p class=\"text-xs text-gray-500\">Version</p>
                      <p id=\"hVersion\" class=\"mt-1 text-sm font-medium\">\u2014</p>
                    </div>
                  </div>
                  <div class=\"mt-6 rounded-lg border p-4\">
                    <p class=\"text-sm font-medium\">Detailed Status</p>
                    <pre id=\"statusJson\" class=\"mt-2 bg-gray-50 rounded p-3 text-xs overflow-x-auto\"></pre>
                  </div>
                </div>
              </section>

              <section>
                <div class=\"rounded-xl border bg-white p-6 shadow-sm\">
                  <h2 class=\"text-base font-semibold\">Paste Cookie JSON</h2>
                  <p class=\"mt-1 text-xs text-gray-500\">Only __Secure-1PSID and __Secure-1PSIDTS are stored.</p>
                  <form id=\"cookieForm\" class=\"mt-4 space-y-3\">
                    {token_field}
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
        async function loadHealth() {{
          try {{
            const h = await fetch('/health').then(r => r.json());
            document.getElementById('hStatus').textContent = h.status;
            document.getElementById('hClient').textContent = String(h.client_ready);
            document.getElementById('hSessions').textContent = String(h.active_sessions);
            document.getElementById('hVersion').textContent = h.version || '1.0.0';
          }} catch(e) {{}}
          try {{
            const s = await fetch('/status').then(r => r.json());
            document.getElementById('statusJson').textContent = JSON.stringify(s, null, 2);
          }} catch(e) {{}}
        }}
        document.getElementById('refreshBtn').addEventListener('click', loadHealth);
        loadHealth();

        document.getElementById('cookieForm').addEventListener('submit', async (ev) => {{
          ev.preventDefault();
          const el = document.getElementById('cookieMsg');
          el.textContent = '';
          const fd = new FormData(ev.target);
          try {{
            const res = await fetch('/cookies', {{ method: 'POST', body: fd }});
            const data = await res.json();
            if (data && data.success) {{
              el.textContent = 'Cookies updated successfully';
              el.className = 'mt-3 text-sm text-green-600';
              loadHealth();
            }} else {{
              el.textContent = (data && data.error) ? data.error : 'Update failed';
              el.className = 'mt-3 text-sm text-red-600';
            }}
          }} catch(e) {{
            el.textContent = String(e);
            el.className = 'mt-3 text-sm text-red-600';
          }}
        }});
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
            kw = {"model": model}
            if request.system_prompt:
                kw["system_prompt"] = request.system_prompt
            response = await client.generate_content(
                prompt=request.message,
                **kw
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
            kw = {"model": model}
            if request.system_prompt:
                kw["system_prompt"] = request.system_prompt
            response = await chat.send_message(
                prompt=request.message,
                **kw
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

@app.get("/gems")
async def list_gems():
    """Scan/list user's Gems and merge with stored records."""
    try:
        stored = _load_gems_from_store()
        generated: List[Dict[str, Any]] = []
        try:
            client = await get_or_init_client()
            response = await client.generate_content(
                prompt='''List ALL my custom Gems from gemini.google.com. Output ONLY valid JSON array of objects with keys name, id, desc.''',
                model=Model.G_2_5_PRO
            )
            import re
            import json as _json
            json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response.text)
            if json_match:
                generated = _json.loads(json_match.group())
        except Exception:
            generated = []
        def norm(items: List[Dict[str, Any]]):
            out = []
            seen = set()
            for it in items:
                gid = (it.get("id") or "").strip()
                if not gid or gid in seen:
                    continue
                seen.add(gid)
                out.append({
                    "name": it.get("name") or "",
                    "id": gid,
                    "desc": it.get("desc") or ""
                })
            return out
        merged = norm(stored) + [x for x in norm(generated) if x["id"] not in {y["id"] for y in norm(stored)}]
        return {"stored": norm(stored), "generated": norm(generated), "merged": merged, "count": len(merged), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.warning(f"Gems list query failed: {e}")
        return {"error": str(e), "gems": [], "note": "Client/auth issue? Update cookies first."}

class GemsPayload(BaseModel):
    gems: List[Dict[str, Any]]

@app.post("/gems")
async def import_gems(payload: GemsPayload, token: Optional[str] = Form(default=None)):
    if COOKIE_UPDATE_TOKEN and token is not None and token != COOKIE_UPDATE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    gems = payload.gems if isinstance(payload.gems, list) else []
    cleaned = []
    seen = set()
    for it in gems:
        gid = (it.get("id") or "").strip()
        if not gid or gid in seen:
            continue
        seen.add(gid)
        cleaned.append({
            "name": it.get("name") or "",
            "id": gid,
            "desc": it.get("desc") or ""
        })
    existing = _load_gems_from_store()
    existing_ids = {x.get("id") for x in existing}
    final = existing + [x for x in cleaned if x["id"] not in existing_ids]
    _persist_gems(final)
    return {"success": True, "count": len(final)}

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
