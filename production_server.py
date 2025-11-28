#!/usr/bin/env python3
"""
Production-ready Gemini API Server for Railway deployment
Optimized for cloud deployment with proper error handling and security
"""

import os
import re
import json
import asyncio
import uuid
import logging
import shutil
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager
from urllib.parse import urlparse

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
clients: Dict[str, GeminiClient] = {}
chat_sessions: Dict[str, Dict[str, Any]] = {}

# Query queue manager to handle rate limiting and race conditions
class QueryQueueManager:
    """Manages Gemini API requests with rate limiting and prevents race conditions"""
    
    def __init__(self, max_concurrent: int = 5, rate_limit_per_minute: int = 60):  # Increased for speed
        self.max_concurrent = max_concurrent
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_queue = asyncio.Queue()
        self.active_requests = 0
        self.request_times = []  # Track request times for rate limiting
        self.processor_task = None
        self.running = False
        
    async def start(self):
        """Start the queue processor"""
        if self.running:
            return
        self.running = True
        self.processor_task = asyncio.create_task(self._process_queue())
        logger.info("Query queue manager started")
        
    async def stop(self):
        """Stop the queue processor"""
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Query queue manager stopped")
        
    async def _process_queue(self):
        """Process queued requests with rate limiting"""
        while self.running:
            try:
                # Wait for a request (longer timeout to avoid missing requests)
                request_item = await asyncio.wait_for(self.request_queue.get(), timeout=5.0)
                
                # Check rate limiting
                now = datetime.utcnow()
                # Remove requests older than 1 minute
                self.request_times = [t for t in self.request_times if (now - t).total_seconds() < 60]
                
                # If we've hit the rate limit, wait
                if len(self.request_times) >= self.rate_limit_per_minute:
                    sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                        continue
                
                # Wait for available slot
                while self.active_requests >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                
                # Process the request
                self.active_requests += 1
                self.request_times.append(now)
                
                asyncio.create_task(self._execute_request(request_item))
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)
                
    async def _execute_request(self, request_item):
        """Execute a single request"""
        future, account_id, prompt, kwargs = request_item
        try:
            # Execute the Gemini request with timeout
            client = await get_or_init_client(account_id)
            result = await asyncio.wait_for(
                client.generate_content(prompt=prompt, **kwargs),
                timeout=50.0  # 50 seconds max for Gemini API
            )
            
            # Set the result for the future
            future.set_result(result)
            
        except asyncio.TimeoutError:
            future.set_exception(Exception("Gemini API timeout"))
            logger.warning(f"Gemini API timeout for account {account_id}")
        except Exception as e:
            # Set the exception for the future
            future.set_exception(e)
            logger.error(f"Gemini API error for account {account_id}: {e}")
        finally:
            self.active_requests -= 1
            
    async def submit_request(self, account_id: str, prompt: str, **kwargs):
        """Submit a request to the queue and return the result"""
        if not self.running:
            await self.start()
            
        # Create a future for the result
        future = asyncio.Future()
        
        # Add to queue with timeout (30 seconds max wait to get into queue)
        try:
            await asyncio.wait_for(
                self.request_queue.put((future, account_id, prompt, kwargs)),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            future.set_exception(Exception("Queue full, please try again later"))
            raise HTTPException(status_code=503, detail="Service temporarily busy, please try again later")
        
        # Wait for the result with timeout (60 seconds max for API call)
        try:
            return await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout, please try again")

# Global queue manager
queue_manager = QueryQueueManager()

# Simple cookie refresh scheduler to avoid circular imports
class CookieRefreshScheduler:
    """Background task to refresh cookies proactively"""
    
    def __init__(self, refresh_interval_minutes: int = 9):
        self.refresh_interval = refresh_interval_minutes * 60  # Convert to seconds
        self.running = False
        self.task = None
        self.last_refresh_time = datetime.utcnow()
        self.refresh_lock = asyncio.Lock()
        
    async def start(self):
        """Start the background refresh scheduler"""
        if self.running:
            logger.warning("Cookie refresh scheduler already running")
            return
            
        self.running = True
        self.task = asyncio.create_task(self._refresh_loop())
        logger.info(f"Started cookie refresh scheduler (every {self.refresh_interval // 60} minutes)")
        
    async def stop(self):
        """Stop the background refresh scheduler"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped cookie refresh scheduler")
        
    async def _reset_refresh_timer(self):
        """Reset the refresh timer to delay next refresh"""
        async with self.refresh_lock:
            self.last_refresh_time = datetime.utcnow()
            logger.debug(f"Refresh timer reset at {self.last_refresh_time.isoformat()}")
    
    async def _refresh_loop(self):
        """Main refresh loop that runs periodically"""
        while self.running:
            try:
                # Calculate time until next refresh
                time_since_refresh = (datetime.utcnow() - self.last_refresh_time).total_seconds()
                time_until_refresh = max(0, self.refresh_interval - time_since_refresh)
                
                # Wait until next refresh time
                await asyncio.sleep(time_until_refresh)
                if not self.running:
                    break
                    
                await self._refresh_all_clients()
                self.last_refresh_time = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cookie refresh loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def _load_refresh_session(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Load refresh session data for a specific account"""
        sessions = _load_refresh_sessions()
        return sessions.get(account_id)
    
    def _save_refresh_session(self, account_id: str, session_id: str, chat: Any) -> None:
        """Save refresh session data for a specific account"""
        sessions = _load_refresh_sessions()
        sessions[account_id] = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat()
        }
        _save_refresh_sessions(sessions)
    
    def _is_session_too_old(self, session_data: Dict[str, Any]) -> bool:
        """Check if session is older than 24 hours"""
        if not session_data or "created_at" not in session_data:
            return True
        
        try:
            created_at = datetime.fromisoformat(session_data["created_at"])
            age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
            return age_hours > 24
        except (ValueError, KeyError):
            return True
    
    def _get_minimal_question(self) -> str:
        """Generate ultra-minimal math question"""
        import random
        operations = [
            (random.randint(1, 10), "+", random.randint(1, 10)),
            (random.randint(1, 10), "×", random.randint(1, 10)),
        ]
        a, op, b = random.choice(operations)
        result = a + b if op == "+" else a * b
        return f"{a}{op}{b}={result}?"
                
    async def _refresh_all_clients(self):
        """Refresh all active Gemini clients using single persistent thread"""
        if not clients:
            logger.info("No clients to refresh")
            return
            
        refresh_count = 0
        for account_id, client in list(clients.items()):
            try:
                # Load or create refresh session
                session_data = self._load_refresh_session(account_id)
                
                # Check if session is too old (>24h) or doesn't exist
                if not session_data or self._is_session_too_old(session_data):
                    # Create new session
                    chat = client.start_chat()
                    session_id = str(uuid.uuid4())
                    
                    # Store in memory for future reuse
                    if account_id not in chat_sessions:
                        chat_sessions[account_id] = {}
                    chat_sessions[account_id][session_id] = chat
                    
                    # Save to persistent storage
                    self._save_refresh_session(account_id, session_id, chat)
                    logger.info(f"Created new refresh session for account: {account_id}")
                else:
                    # Reuse existing session
                    session_id = session_data["session_id"]
                    if session_id not in chat_sessions.get(account_id, {}):
                        chat_sessions.setdefault(account_id, {})[session_id] = client.start_chat()
                    chat = chat_sessions[account_id][session_id]
                
                # Send ultra-minimal question to refresh cookies
                question = self._get_minimal_question()
                await chat.send_message(question)
                
                # Update last used time
                sessions = _load_refresh_sessions()
                if account_id in sessions:
                    sessions[account_id]["last_used"] = datetime.utcnow().isoformat()
                    _save_refresh_sessions(sessions)
                
                refresh_count += 1
                logger.info(f"Refreshed cookies for account: {account_id}")
                
            except Exception as e:
                logger.error(f"Failed to refresh cookies for account {account_id}: {e}")
                
        if refresh_count > 0:
            logger.info(f"Successfully refreshed {refresh_count} account(s) at {datetime.utcnow().isoformat()}")

# Global scheduler instance
scheduler = CookieRefreshScheduler()

# Production environment variables
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
COOKIE_UPDATE_TOKEN = os.getenv("COOKIE_UPDATE_TOKEN")

STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "/session-cookie"))
try:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    raise RuntimeError(f"Storage root unavailable: {exc}")

COOKIES_FILE = Path(os.getenv("COOKIE_STORE_PATH", str(STORAGE_ROOT / "cookies.json")))
GEMS_FILE = Path(os.getenv("GEMS_STORE_PATH", str(STORAGE_ROOT / "gems.json")))
REFRESH_SESSIONS_FILE = Path(os.getenv("REFRESH_SESSIONS_PATH", str(STORAGE_ROOT / "refresh_sessions.json")))


def _ensure_writable_dir(dir_path: Path) -> None:
    """
    Ensure the target directory exists and is writable.
    If permissions are too strict (e.g., volume mounted root-only), try to relax them once.
    Hard-fail with a clear error if still not writable.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    test_file = dir_path / ".perm_test"
    try:
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return
    except Exception:
        pass

    try:
        dir_path.chmod(0o777)
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return
    except Exception as exc:
        raise RuntimeError(
            f"Storage path not writable: {dir_path}. "
            f"Fix volume permissions or mount path (consider RAILWAY_RUN_UID=0 or pre-start chmod). Error: {exc}"
        )


_ensure_writable_dir(STORAGE_ROOT)

# Pydantic models for production
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Message to send to Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt or full gem:// URL (e.g. from gemini.google.com/gem/ID share link). Create/edit Gems on web.")
    account_id: Optional[str] = Field(None, description="Account ID to use (default: primary)")
    include_thoughts: bool = Field(default=False, description="Include verbose thoughts in response (slower)")

class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: Optional[str] = None
    candidates_count: int
    thoughts: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
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
    
    # Start the cookie refresh scheduler
    try:
        await scheduler.start()
        logger.info("Cookie refresh scheduler started")
    except Exception as e:
        logger.error(f"Failed to start cookie refresh scheduler: {e}")
    
    # Start the query queue manager
    try:
        await queue_manager.start()
        logger.info("Query queue manager started")
    except Exception as e:
        logger.error(f"Failed to start query queue manager: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gemini API Server")
    
    # Stop the scheduler first
    try:
        await scheduler.stop()
        logger.info("Cookie refresh scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping cookie refresh scheduler: {e}")
    
    # Stop the queue manager
    try:
        await queue_manager.stop()
        logger.info("Query queue manager stopped")
    except Exception as e:
        logger.error(f"Error stopping query queue manager: {e}")
    
    # Close all clients
    for _id, cli in list(clients.items()):
        try:
            await cli.close()
        except Exception:
            pass

def _account_dir(account_id: str) -> Path:
    return STORAGE_ROOT / "accounts" / account_id

def _cookies_path_for(account_id: str) -> Path:
    return _account_dir(account_id) / "cookies.json"

def _gems_path_for(account_id: str) -> Path:
    return _account_dir(account_id) / "gems.json"

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


def _load_refresh_sessions() -> Dict[str, Dict[str, Any]]:
    """Load refresh session data from storage"""
    try:
        if REFRESH_SESSIONS_FILE.exists():
            data = json.loads(REFRESH_SESSIONS_FILE.read_text())
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read refresh sessions: {exc}")
    return {}


def _save_refresh_sessions(sessions: Dict[str, Dict[str, Any]]) -> None:
    """Save refresh session data to storage"""
    try:
        REFRESH_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        REFRESH_SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))
    except Exception as exc:
        logger.error(f"Failed to save refresh sessions: {exc}")


_GEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _extract_gem_id(system_prompt: str) -> Optional[str]:
    """
    Normalize system_prompt into a gem id. Accepts:
    - gem://<id> or gemini://<id>
    - https://gemini.google.com/gem/<id> (or /app/gem/<id>)
    - bare gem id
    Returns the gem id or None if it cannot be parsed.
    """
    if not system_prompt:
        return None

    sp = system_prompt.strip()
    if not sp:
        return None

    lower = sp.lower()
    for prefix in ("gem://", "gemini://"):
        if lower.startswith(prefix):
            candidate = sp[len(prefix):].strip().strip("/")
            return candidate or None

    if "gemini.google.com" in lower:
        parsed = urlparse(sp)
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            candidate = parts[-1].split("?")[0].split("#")[0].strip()
            if _GEM_ID_PATTERN.match(candidate):
                return candidate

    if _GEM_ID_PATTERN.match(sp):
        return sp

    return None


def _build_prompt_kwargs(model: Model, system_prompt: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Build keyword arguments for Gemini calls while validating system_prompt as a Gem.
    Returns (kwargs, gem_id) so chat sessions can persist the Gem choice.
    """
    kw: Dict[str, Any] = {"model": model}
    gem_id = None
    if system_prompt:
        gem_id = _extract_gem_id(system_prompt)
        if not gem_id:
            raise HTTPException(
                status_code=400,
                detail="system_prompt must be a gem://<id> or Gemini gem share URL"
            )
        kw["gem"] = gem_id
    return kw, gem_id


def _get_stored_credentials_for(account_id: str) -> Tuple[Optional[str], Optional[str]]:
    if account_id == "primary":
        env_psid = os.getenv("SECURE_1PSID")
        env_psidts = os.getenv("SECURE_1PSIDTS")
        if env_psid:
            return env_psid, env_psidts
        return _load_cookies_from_store()
    try:
        p = _cookies_path_for(account_id)
        if p.exists():
            data = json.loads(p.read_text())
            return data.get("SECURE_1PSID"), data.get("SECURE_1PSIDTS")
    except Exception:
        pass
    return None, None


async def get_or_init_client(account_id: Optional[str] = None, force_reload: bool = False) -> GeminiClient:
    account = (account_id or "primary").strip() or "primary"
    if force_reload and account in clients:
        try:
            await clients[account].close()
        except Exception:
            pass
        clients.pop(account, None)
    if account not in clients:
        secure_1psid, secure_1psidts = _get_stored_credentials_for(account)
        if not secure_1psid:
            raise ValueError("SECURE_1PSID environment variable is required for this account")
        try:
            cli = GeminiClient(secure_1psid, secure_1psidts)
            await cli.init(auto_refresh=True)
            clients[account] = cli
            if account not in chat_sessions:
                chat_sessions[account] = {}
        except AuthError as e:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable - authentication failed")
        except Exception:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    return clients[account]

def get_model_enum(model_name: str) -> Model:
    """Convert model name string to Model enum with validation"""
    model_map = {
        "gemini-3.0-pro": Model.G_3_0_PRO,
        "gemini-2.5-pro": Model.G_2_5_PRO,
        "gemini-2.5-flash": Model.G_2_5_FLASH,
        "unspecified": Model.UNSPECIFIED,
    }
    return model_map.get(model_name, Model.G_2_5_FLASH)


def _extract_secure_cookies_any(cookie_payload: Any, raw_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cookies: List[Any] = []
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
        if isinstance(entry, dict):
            name = entry.get("name")
            value = entry.get("value")
            if name == "__Secure-1PSID":
                secure_1psid = value
            elif name == "__Secure-1PSIDTS":
                secure_1psidts = value

    if (not secure_1psid) and isinstance(raw_text, str) and raw_text:
        try:
            import re
            m = re.search(r"""name"\s*:\s*"__Secure-1PSID"[\s\S]*?"value"\s*:\s*"([^"]+)""", raw_text)
            if m:
                secure_1psid = m.group(1)
            m2 = re.search(r"""name"\s*:\s*"__Secure-1PSIDTS"[\s\S]*?"value"\s*:\s*"([^"]+)""", raw_text)
            if m2:
                secure_1psidts = m2.group(1)
        except Exception:
            pass

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

    html_content = """
    <!doctype html>
    <html lang=\"en">
      <head>
        <meta charset=\"utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Gemini API Server</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
        <style>
          body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
        </style>
      </head>
      <body class="bg-gray-50">
        <div class="min-h-screen">
          <header class="bg-white border-b">
            <div class="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
              <div class="flex items-center gap-2">
                <div class="h-8 w-8 rounded-lg bg-indigo-600"></div>
                <div>
                  <h1 class="text-lg font-semibold">Gemini API Server</h1>
                  <p class="text-xs text-gray-500">Environment: __ENV__</p>
                </div>
              </div>
              <div class="flex items-center gap-2">
                <a href="/docs" class="inline-flex items-center rounded-md bg-gray-900 text-white text-sm px-3 py-2 hover:bg-black">API Docs</a>
                <a href="/redoc" class="inline-flex items-center rounded-md bg-white border text-sm px-3 py-2 hover:bg-gray-100">ReDoc</a>
              </div>
            </div>
          </header>

          <main class="max-w-5xl mx-auto px-4 py-8">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <section class="lg:col-span-2">
                <div class="rounded-xl border bg-white p-6 shadow-sm">
                  <div class="flex items-center justify-between">
                    <h2 class="text-base font-semibold">Server Health</h2>
                    <button id="refreshBtn" class="text-sm px-3 py-1.5 rounded-md border hover:bg-gray-50">Refresh</button>
                  </div>
                  <div id="health" class="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div class="rounded-lg border p-4">
                      <p class="text-xs text-gray-500">Status</p>
                      <p id="hStatus" class="mt-1 text-sm font-medium">—</p>
                    </div>
                    <div class="rounded-lg border p-4">
                      <p class="text-xs text-gray-500">Client Ready</p>
                      <p id="hClient" class="mt-1 text-sm font-medium">—</p>
                    </div>
                    <div class="rounded-lg border p-4">
                      <p class="text-xs text-gray-500">Active Sessions</p>
                      <p id="hSessions" class="mt-1 text-sm font-medium">—</p>
                    </div>
                    <div class="rounded-lg border p-4">
                      <p class="text-xs text-gray-500">Version</p>
                      <p id="hVersion" class="mt-1 text-sm font-medium">—</p>
                    </div>
                  </div>
                  <div class="mt-6 rounded-lg border bg-white p-6 shadow-sm">
                    <div class="flex items-center justify-between">
                        <h2 class="text-base font-semibold">Accounts</h2>
                        <button id="refreshAccountsBtn" class="text-sm px-3 py-1.5 rounded-md border hover:bg-gray-50">Refresh</button>
                    </div>
                    <div id="accounts-cards-container" class="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        <!-- Account cards will be dynamically inserted here -->
                    </div>
                  </div>
                  <div class="mt-6 rounded-lg border p-4">
                    <p class="text-sm font-medium">Detailed Status</p>
                    <pre id="statusJson" class="mt-2 bg-gray-50 rounded p-3 text-xs overflow-x-auto"></pre>
                  </div>
                </div>
              </section>

              <section>
                <div class="rounded-xl border bg-white p-6 shadow-sm">
                  <h2 class="text-base font-semibold">Paste Cookie JSON</h2>
                  <p class="mt-1 text-xs text-gray-500">Only __Secure-1PSID and __Secure-1PSIDTS are stored.</p>
                  <form id="cookieForm" method="post" action="/cookies" enctype="multipart/form-data" class="mt-4 space-y-3">
                    __TOKEN__
                    <label class="block text-sm font-medium text-gray-700">Account ID</label>
                    <input type="text" id="account_id" name="account_id" placeholder="primary" class="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500" />
                    <label class="block text-sm font-medium text-gray-700">Cookie JSON</label>
                    <textarea name="cookie_json" id="cookie_json" class="mt-1 w-full h-40 rounded-md border border-gray-300 px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="[{{}}]" required></textarea>
                    <button type="submit" class="w-full inline-flex items-center justify-center rounded-md bg-indigo-600 text-white text-sm px-3 py-2 hover:bg-indigo-700">Update Cookies</button>
                  </form>
                  <div id="cookieMsg" class="mt-3 text-sm"></div>
                </div>
                <div class="mt-6 rounded-xl border bg-white p-6 shadow-sm">
                  <div class="flex items-center justify-between">
                    <h2 class="text-base font-semibold">My Gems</h2>
                    <button id="scanGemsBtn" class="text-sm px-3 py-1.5 rounded-md border hover:bg-gray-50">Scan Gems</button>
                  </div>
                  <div class="mt-3 grid grid-cols-2 gap-3">
                    <div class="rounded-lg border p-3">
                      <p class="text-xs text-gray-500">Stored</p>
                      <p id="gStored" class="mt-1 text-sm font-medium">0</p>
                    </div>
                    <div class="rounded-lg border p-3">
                      <p class="text-xs text-gray-500">Merged</p>
                      <p id="gMerged" class="mt-1 text-sm font-medium">0</p>
                    </div>
                  </div>
                  <div class="mt-3">
                    <ul id="gemsList" class="space-y-2"></ul>
                  </div>
                  <div class="mt-4">
                    <label class="block text-sm font-medium text-gray-700">Import Gems JSON</label>
                    <div class="space-y-2">
                      <div class="flex gap-2">
                        <input id="gem_url_input" type="text" class="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Gem URL or ID (e.g. https://gemini.google.com/gem/abc123)" />
                        <input id="gem_name_input" type="text" class="w-48 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Name (optional)" />
                        <button id="addGemBtn" type="button" class="rounded-md bg-gray-900 text-white text-sm px-3 py-2 hover:bg-black">Add</button>
                      </div>
                      <textarea id="gems_json" class="mt-1 w-full h-32 rounded-md border border-gray-300 px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder='[{"name":"Gem Name","id":"https://gemini.google.com/gem/ID"}]'></textarea>
                      <button id="importGemsBtn" class="w-full inline-flex items-center justify-center rounded-md bg-indigo-600 text-white text-sm px-3 py-2 hover:bg-indigo-700">Import Gems</button>
                      <div id="gemsMsg" class="mt-2 text-sm"></div>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </main>

          <footer class="mt-8 py-6 text-center text-xs text-gray-500">Gemini API Server</footer>
        </div>

        <script>
        async function loadHealth() {
          try {
            const h = await fetch('/health').then(r => r.json());
            document.getElementById('hStatus').textContent = h.status;
            document.getElementById('hClient').textContent = String(h.client_ready);
            document.getElementById('hSessions').textContent = String(h.active_sessions);
            document.getElementById('hVersion').textContent = h.version || '1.0.0';
          } catch(e) {}
          try {
            const s = await fetch('/status').then(r => r.json());
            document.getElementById('statusJson').textContent = JSON.stringify(s, null, 2);
          } catch(e) {}
        }
        document.getElementById('refreshBtn').addEventListener('click', loadHealth);
        loadHealth();

        async function loadAccounts() {
          const container = document.getElementById('accounts-cards-container');
          container.innerHTML = '';
          let accounts = [];
          try {
            const a = await fetch('/status/accounts');
            if (a.ok) {
              const data = await a.json();
              accounts = data.accounts || [];
            }
          } catch (e) {}
          if (!accounts.length) {
            container.innerHTML = '<p class="text-sm text-gray-500">No accounts found.</p>';
            return;
          }
          accounts.forEach(acc => {
            const card = document.createElement('div');
            card.className = 'rounded-lg border bg-white p-4 shadow-sm';
            
            const header = document.createElement('div');
            header.className = 'flex items-center justify-between';
            
            const accountId = document.createElement('p');
            accountId.className = 'text-sm font-medium text-gray-900';
            accountId.textContent = acc.account_id || 'N/A';
            
            const statusDot = document.createElement('div');
            const statusColor = acc.ready ? 'bg-green-500' : (acc.error ? 'bg-red-500' : 'bg-yellow-500');
            statusDot.className = `h-2.5 w-2.5 rounded-full ${statusColor}`;
            
            header.appendChild(accountId);
            header.appendChild(statusDot);
            
            const statusText = document.createElement('p');
            statusText.className = 'mt-2 text-xs text-gray-500';
            const status = acc.ready ? 'Ready' : (acc.error ? `Error: ${acc.error}` : 'Not Ready');
            statusText.textContent = status;
            
            const cookieInfo = document.createElement('p');
            cookieInfo.className = 'mt-1 text-xs text-gray-500';
            const hasP = acc.cookies ? (acc.cookies.has_psid ? 'PSID: Yes' : 'PSID: No') : '';
            const hasT = acc.cookies ? (acc.cookies.has_psidts ? 'SIDTS: Yes' : 'SIDTS: No') : '';
            cookieInfo.textContent = `${hasP}, ${hasT}`;

            card.appendChild(header);
            card.appendChild(statusText);
            card.appendChild(cookieInfo);
            
            container.appendChild(card);
          });
        }
        document.getElementById('refreshAccountsBtn').addEventListener('click', loadAccounts);
        loadAccounts();

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
              loadAccounts();
            } else {
              el.textContent = (data && data.error) ? data.error : 'Update failed';
              el.className = 'mt-3 text-sm text-red-600';
            }
          } catch(e) {
            el.textContent = String(e);
            el.className = 'mt-3 text-sm text-red-600';
          }
        });

        async function renderGems(data) {
          try {
            const stored = (data.stored || []);
            const merged = (data.merged || []);
            document.getElementById('gStored').textContent = String(stored.length);
            document.getElementById('gMerged').textContent = String(merged.length);
            const ul = document.getElementById('gemsList');
            ul.innerHTML = '';
            merged.forEach(g => {
              const li = document.createElement('li');
              li.className = 'rounded-lg border p-3 flex items-center justify-between';
              const left = document.createElement('div');
              const a = document.createElement('a');
              a.href = g.id;
              a.textContent = g.name || g.id;
              a.target = '_blank';
              a.className = 'text-sm font-medium text-indigo-600 hover:underline';
              const small = document.createElement('p');
              small.className = 'text-xs text-gray-500';
              small.textContent = g.id;
              left.appendChild(a);
              left.appendChild(small);
              const btn = document.createElement('button');
              btn.className = 'text-xs px-2 py-1 rounded-md border hover:bg-gray-50';
              btn.textContent = 'Copy ID';
              btn.addEventListener('click', async () => {
                try { await navigator.clipboard.writeText(g.id); } catch(e) {}
              });
              li.appendChild(left);
              li.appendChild(btn);
              ul.appendChild(li);
            });
          } catch(e) {}
        }

        async function loadGems() {
          try {
            const res = await fetch('/gems');
            const data = await res.json();
            await renderGems(data);
          } catch(e) {}
        }
        document.getElementById('scanGemsBtn').addEventListener('click', loadGems);
        loadGems();

        document.getElementById('importGemsBtn').addEventListener('click', async () => {
          const el = document.getElementById('gemsMsg');
          el.textContent = '';
          try {
            const txt = document.getElementById('gems_json').value || '[]';
            const arr = JSON.parse(txt);
            const res = await fetch('/gems', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gems: arr }) });
            const data = await res.json();
            if (data && data.success) {
              el.textContent = 'Gems imported';
              el.className = 'mt-2 text-sm text-green-600';
              loadGems();
            } else {
              el.textContent = (data && data.error) ? data.error : 'Import failed';
              el.className = 'mt-2 text-sm text-red-600';
            }
          } catch(e) {
            el.textContent = String(e);
            el.className = 'mt-2 text-sm text-red-600';
          }
        });

        function normalizeGemId(raw) {
          if (!raw) return '';
          const trimmed = raw.trim();
          if (!trimmed) return '';
          if (trimmed.startsWith('http')) return trimmed;
          return `https://gemini.google.com/gem/${trimmed}`;
        }

        document.getElementById('addGemBtn').addEventListener('click', () => {
          const urlRaw = document.getElementById('gem_url_input').value;
          const nameRaw = document.getElementById('gem_name_input').value;
          const id = normalizeGemId(urlRaw);
          const msgEl = document.getElementById('gemsMsg');
          if (!id) {
            msgEl.textContent = 'Please enter a Gem URL or ID.';
            msgEl.className = 'mt-2 text-sm text-red-600';
            return;
          }
          let arr = [];
          const textarea = document.getElementById('gems_json');
          if (textarea.value.trim()) {
            try { arr = JSON.parse(textarea.value); } catch (e) { arr = []; }
            if (!Array.isArray(arr)) arr = [];
          }
          arr.push({ name: nameRaw || id.split('/').pop(), id });
          textarea.value = JSON.stringify(arr, null, 2);
          document.getElementById('gem_url_input').value = '';
          document.getElementById('gem_name_input').value = '';
          msgEl.textContent = 'Added to import list.';
          msgEl.className = 'mt-2 text-sm text-green-600';
        });
        </script>
      </body>
    </html>
    """.replace("__ENV__", RAILWAY_ENVIRONMENT).replace("__TOKEN__", token_field)
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
            <p class="message">Paste the export from your browser's cookie viewer. Only __Secure-1PSID and __Secure-1PSIDTS are stored.</p>
            <form method="post" action="/cookies">
                {token_field}
                <textarea name="cookie_json" placeholder="[{{}}]" required></textarea>
                <button type="submit">Update Cookies</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/cookies")
async def update_cookies(cookie_json: str = Form(...), account_id: Optional[str] = Form(default="primary"), token: Optional[str] = Form(default=None)):
    """Accept cookie JSON, persist it, and reload the Gemini client."""
    if COOKIE_UPDATE_TOKEN and token != COOKIE_UPDATE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    raw_text = cookie_json
    aid = (account_id or "primary").strip() or "primary"
    try:
        payload = json.loads(cookie_json)
    except json.JSONDecodeError:
        payload = {"cookies": []}

    secure_1psid, secure_1psidts = _extract_secure_cookies_any(payload, raw_text)

    if not secure_1psid:
        raise HTTPException(status_code=400, detail="__Secure-1PSID cookie not found")

    if aid == "primary":
        os.environ["SECURE_1PSID"] = secure_1psid
        if secure_1psidts:
            os.environ["SECURE_1PSIDTS"] = secure_1psidts
        _persist_cookies(secure_1psid, secure_1psidts)
    else:
        p = _cookies_path_for(aid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"SECURE_1PSID": secure_1psid, "SECURE_1PSIDTS": secure_1psidts}))

    await get_or_init_client(aid, force_reload=True)

    return {
        "success": True,
        "message": "Gemini cookies updated successfully",
        "updated_at": datetime.utcnow().isoformat(),
        "has_sidts": bool(secure_1psidts),
        "account_id": aid
    }

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Production health check for Railway monitoring"""
    try:
        client_ready = len(clients) > 0
        return StatusResponse(
            status="healthy" if client_ready else "initializing",
            environment=RAILWAY_ENVIRONMENT,
            client_ready=client_ready,
            active_sessions=sum(len(v) for v in chat_sessions.values()),
            uptime="running"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/scheduler/health")
async def scheduler_health():
    """Dedicated endpoint to check cookie refresh scheduler status"""
    try:
        return {
            "scheduler_running": scheduler.running,
            "refresh_interval_minutes": scheduler.refresh_interval // 60,
            "active_clients": len(clients),
            "last_refresh": scheduler.last_refresh_time.isoformat() if scheduler.running else "never",
            "status": "active" if scheduler.running else "inactive"
        }
    except Exception as e:
        logger.error(f"Scheduler health check failed: {e}")
        return {"error": str(e), "status": "unknown"}

@app.get("/queue/status")
async def queue_status():
    """Check queue manager status"""
    try:
        return {
            "queue_running": queue_manager.running,
            "active_requests": queue_manager.active_requests,
            "queue_size": queue_manager.request_queue.qsize(),
            "max_concurrent": queue_manager.max_concurrent,
            "rate_limit_per_minute": queue_manager.rate_limit_per_minute,
            "requests_in_last_minute": len(queue_manager.request_times),
            "timeout_settings": {
                "queue_timeout": 30,  # seconds to get into queue
                "request_timeout": 60,  # seconds for total request
                "gemini_timeout": 50,  # seconds for Gemini API call
                "processor_timeout": 5  # seconds for queue processor
            },
            "status": "active" if queue_manager.running else "inactive"
        }
    except Exception as e:
        logger.error(f"Queue status check failed: {e}")
        return {"error": str(e), "status": "unknown"}

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
        account_id = request.account_id or "primary"
        model = get_model_enum(request.model)
        
        # Build prompt kwargs
        kw, _ = _build_prompt_kwargs(model, request.system_prompt)
        
        try:
            # Use the queue manager to send the request
            response = await queue_manager.submit_request(
                account_id=account_id,
                prompt=request.message,
                **kw
            )
            
            # Reset the refresh timer on successful response
            await scheduler._reset_refresh_timer()
            
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
        
        # Build response without thoughts by default for speed
        chat_response = ChatResponse(
            response=response.text,
            model=model.model_name,
            candidates_count=len(response.candidates),
            thoughts=response.thoughts if request.include_thoughts else None,
            images=images,
            metadata={
                "rcid": response.rcid,
                "metadata_length": len(response.metadata)
            }
        )
        
        # Include raw data only if explicitly requested
        if request.include_thoughts:
            chat_response.raw = {
                "response": response.text,
                "model": model.model_name,
                "candidates_count": len(response.candidates),
                "thoughts": response.thoughts,
                "images": images,
                "metadata": {
                    "rcid": response.rcid,
                    "chat_metadata_length": len(response.metadata)
                }
            }
        
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in send_message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def send_chat_message(session_id: str, request: MessageRequest):
    """Send message in chat session with session management"""
    try:
        acc = (request.account_id or "primary").strip() or "primary"
        client = await get_or_init_client(acc)
        
        # Get or create chat session
        if acc not in chat_sessions:
            chat_sessions[acc] = {}
        if session_id not in chat_sessions[acc]:
            chat_sessions[acc][session_id] = client.start_chat()
        chat = chat_sessions[acc][session_id]
        model = get_model_enum(request.model)
        
        try:
            kw, gem_id = _build_prompt_kwargs(model, request.system_prompt)
            chat.model = kw["model"]
            if gem_id:
                chat.gem = gem_id
            
            # Use the queue manager for session messages as well
            # Note: For chat sessions, we need to handle this differently
            # since we can't directly use the queue manager with chat sessions
            response = await chat.send_message(prompt=request.message)
            
            # Reset the refresh timer on successful response
            await scheduler._reset_refresh_timer()
            
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
        
        # Build response without thoughts by default for speed
        chat_response = ChatResponse(
            response=response.text,
            model=model.model_name,
            session_id=session_id,
            candidates_count=len(response.candidates),
            thoughts=response.thoughts if request.include_thoughts else None,
            images=images,
            metadata={
                "rcid": response.rcid,
                "chat_metadata_length": len(chat.metadata)
            }
        )
        
        # Include raw data only if explicitly requested
        if request.include_thoughts:
            chat_response.raw = {
                "response": response.text,
                "model": model.model_name,
                "session_id": session_id,
                "candidates_count": len(response.candidates),
                "thoughts": response.thoughts,
                "images": images,
                "metadata": {
                    "rcid": response.rcid,
                    "chat_metadata_length": len(chat.metadata)
                }
            }
        
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def get_shared_gem(gem_id: str):
    """Fetches a shared gem by its ID and returns its metadata."""
    if not gem_id or not _GEM_ID_PATTERN.match(gem_id):
        raise HTTPException(status_code=400, detail="Invalid gem_id format")
    
    url = f"https://gemini.google.com/gem/{gem_id}"
    
    try:
        # Use any available client to make the HTTP request
        client = await get_or_init_client()
        
        async with client.client.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail="Failed to fetch gem data")
            
            html_content = await response.text()
            
            # Use BeautifulSoup to parse the HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract the data from the script tag
            script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
            if not script_tag:
                raise HTTPException(status_code=500, detail="Failed to find gem data in page")
                
            next_data = json.loads(script_tag.string)
            gem_data = next_data.get('props', {}).get('pageProps', {}).get('gem')
            
            if not gem_data:
                raise HTTPException(status_code=404, detail="Gem not found or not public")

            return {
                "id": gem_data.get('id'),
                "name": gem_data.get('name'),
                "description": gem_data.get('description'),
                "author": gem_data.get('author'),
                "public": gem_data.get('public'),
                "predefined": gem_data.get('predefined'),
                "url": url
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching shared gem: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching gem")


@app.post("/chat/new", response_model=Dict[str, str])
async def create_chat_session():
    """Create new chat session with unique ID"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "status": "created"}

@app.get("/gems/shared/{gem_id}")
async def get_shared_gem_endpoint(gem_id: str):
    """Endpoint to fetch a shared gem by its ID."""
    try:
        gem_data = await get_shared_gem(gem_id)
        return gem_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get shared gem: {e}")
        raise HTTPException(status_code=500, detail="Failed to get shared gem")

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
            jar = await client.fetch_gems(include_hidden=False)
            for gem in jar:
                generated.append({
                    "name": gem.name or gem.id,
                    "id": f"https://gemini.google.com/gem/{gem.id}",
                    "gem_id": gem.id,
                    "desc": gem.description or "",
                    "predefined": gem.predefined,
                })
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
                    "gem_id": it.get("gem_id") or "",
                    "desc": it.get("desc") or "",
                    "predefined": bool(it.get("predefined", False))
                })
            return out

        norm_stored = norm(stored)
        norm_generated = norm(generated)
        merged = norm_stored + [x for x in norm_generated if x["id"] not in {y["id"] for y in norm_stored}]

        return {
            "stored": norm_stored,
            "generated": norm_generated,
            "merged": merged,
            "count": len(merged),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.warning(f"Gems list query failed: {e}")
        return {"error": str(e), "gems": [], "note": "Client/auth issue? Update cookies first."}

class GemsPayload(BaseModel):
    gems: List[Dict[str, Any]]

@app.post("/gems")
async def import_gems(payload: GemsPayload, token: Optional[str] = None):
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
    """Detailed status for monitoring (multi-account aware)"""
    try:
        client_stats = {
            "initialized": len(clients) > 0,
            "active_sessions": sum(len(v) for v in chat_sessions.values()),
            "environment": RAILWAY_ENVIRONMENT,
            "server_uptime": "running",
            "accounts": list(clients.keys()),
            "cookie_scheduler": {
                "running": scheduler.running,
                "refresh_interval_minutes": scheduler.refresh_interval // 60
            }
        }
        # Pick any available client to test
        test_client = None
        if clients:
            # prefer primary, else first
            test_client = clients.get("primary") or next(iter(clients.values()))
        if test_client:
            try:
                await test_client.generate_content(
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

@app.get("/status/accounts")
async def accounts_status():
    """Per-account readiness and cookie presence"""
    try:
        accs = set(clients.keys())
        acc_dir = BASE_DIR / "data" / "accounts"
        if acc_dir.exists():
            for p in acc_dir.iterdir():
                if p.is_dir():
                    accs.add(p.name)
        # always include primary if env has PSID or legacy store
        env_psid = os.getenv("SECURE_1PSID")
        if env_psid:
            accs.add("primary")
        if not accs:
            return {"accounts": []}
        out = []
        for aid in sorted(accs):
            info = {"account_id": aid, "cookies": {"has_psid": False, "has_psidts": False}, "ready": False}
            # check storage
            psid, psidts = _get_stored_credentials_for(aid)
            info["cookies"]["has_psid"] = bool(psid)
            info["cookies"]["has_psidts"] = bool(psidts)
            # try init without forcing reload
            try:
                cli = await get_or_init_client(aid, force_reload=False)
                info["ready"] = True if cli else False
            except Exception as e:
                info["error"] = str(e)
            out.append(info)
        return {"accounts": out, "count": len(out), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/accounts")
async def list_accounts():
    """List known account ids by scanning storage and environment."""
    try:
        accs = set()
        acc_dir = BASE_DIR / "data" / "accounts"
        if acc_dir.exists():
            # include any folder directly under accounts
            for p in acc_dir.iterdir():
                if p.is_dir():
                    # Check if cookies.json exists to ensure it's a valid account directory
                    if (p / "cookies.json").exists():
                        accs.add(p.name)
        # include primary if env or legacy cookies file exists
        if os.getenv("SECURE_1PSID") or COOKIES_FILE.exists():
            accs.add("primary")
        return {"accounts": sorted(accs), "count": len(accs)}
    except Exception as e:
        return {"error": str(e), "accounts": []}

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
class AccountCookies(BaseModel):
    account_id: str
    cookies: List[Dict[str, Any]]

@app.post("/accounts")
async def upsert_account(payload: AccountCookies, token: Optional[str] = None):
    if COOKIE_UPDATE_TOKEN and token is not None and token != COOKIE_UPDATE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    aid = (payload.account_id or "").strip()
    if not aid:
        raise HTTPException(status_code=400, detail="account_id required")
    secure_1psid, secure_1psidts = _extract_secure_cookies_any({"cookies": payload.cookies}, None)
    if not secure_1psid:
        raise HTTPException(status_code=400, detail="__Secure-1PSID cookie not found")
    p = _cookies_path_for(aid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"SECURE_1PSID": secure_1psid, "SECURE_1PSIDTS": secure_1psidts}))
    except Exception as exc:
        logger.error(f"Failed to persist account cookies: {exc}")
        raise HTTPException(status_code=500, detail="Unable to persist account cookies")
    # Attempt to (re)load client for this account
    try:
        await get_or_init_client(aid, force_reload=True)
    except Exception:
        pass
    return {"success": True, "account_id": aid, "has_sidts": bool(secure_1psidts), "updated_at": datetime.utcnow().isoformat()}


@app.delete("/accounts/{account_id}")
async def delete_account(account_id: str):
    aid = (account_id or "").strip()
    if not aid:
        raise HTTPException(status_code=400, detail="account_id required")

    # Close and remove in-memory client/sessions
    if aid in clients:
        try:
            await clients[aid].close()
        except Exception:
            pass
        clients.pop(aid, None)
    chat_sessions.pop(aid, None)

    # Remove stored cookies for this account
    if aid == "primary":
        try:
            if COOKIES_FILE.exists():
                COOKIES_FILE.unlink()
        except Exception:
            pass
    else:
        acc_dir = _account_dir(aid)
        if acc_dir.exists():
            try:
                shutil.rmtree(acc_dir)
            except Exception:
                pass

    return {"success": True, "account_id": aid, "deleted": True}
