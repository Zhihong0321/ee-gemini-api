#!/usr/bin/env python3
"""
Gemini API Server with API Documentation Enabled
Use this for development and testing of the API endpoints
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the production server
import production_server
import uvicorn

def main():
    """Run server with full API documentation enabled"""
    
    # Force development mode for docs
    os.environ["RAILWAY_ENVIRONMENT"] = "development"
    
    # Set cookies for local testing
    os.environ["SECURE_1PSID"] = "g.a0003ghjUxBx1PBzZiPo5g2zzwwFUyxGN3gtsu8hXZzQTMTRMyuSb6jFyAyZq_lrkO_pLBus-gACgYKAW8SARQSFQHGX2Mi4M56oRV31xknPEO1q1NrZxoVAUF8yKrc1oWJHd-GreuXGn54GUg30076"
    os.environ["SECURE_1PSIDTS"] = "sidts-CjEBwQ9iI5gH7XWWJzf4b434kSUbkQl2UJ7ZmEPDr93TJNpUwqEAKVI680C3sK2BWR8AEAA"
    
    print("Starting Gemini API Server with Documentation")
    print("API Docs: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("Health: http://localhost:8000/health")
    print()
    print("Available endpoints:")
    print("- GET  /                    - Root info")
    print("- GET  /health             - Health check")
    print("- GET  /models             - List available models")
    print("- POST /chat               - Send single message")
    print("- POST /chat/new           - Create chat session")
    print("- POST /chat/{session_id}  - Send message to session")
    print("- GET  /chat/{session_id}  - Get session info")
    print("- DEL  /chat/{session_id}  - Delete session")
    print("- GET  /status             - Detailed status")
    print()
    
    # Override server config for local development
    uvicorn.run(
        "production_server:app",
        host="0.0.0.0",  # Bind to all interfaces
        port=8000,
        log_level="info",
        reload=True,  # Enable hot reload for development
        access_log=True
    )

if __name__ == "__main__":
    main()
