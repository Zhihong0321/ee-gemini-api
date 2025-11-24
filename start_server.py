#!/usr/bin/env python3
"""
Startup script for Gemini API Server
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set"""
    # Load .env file
    load_dotenv()
    
    secure_1psid = os.getenv("SECURE_1PSID")
    secure_1psidts = os.getenv("SECURE_1PSIDTS")
    
    print("=== Gemini API Server Startup ===")
    print()
    
    if not secure_1psid:
        print("ERROR: SECURE_1PSID environment variable is not set!")
        print()
        print("To fix this:")
        print("1. Go to https://gemini.google.com and login")
        print("2. Open browser dev tools (F12)")
        print("3. Go to Network tab and refresh")
        print("4. Click any request and copy the __Secure-1PSID cookie")
        print("5. Set environment variable:")
        print("   set SECURE_1PSID=paste_cookie_value_here")
        print("   set SECURE_1PSIDTS=paste_cookie_value_here  # Optional")
        print()
        print("Or create a .env file with these values.")
        return False
    
    print("SECURE_1PSID is configured")
    if secure_1psidts:
        print("SECURE_1PSIDTS is configured")
    else:
        print("WARNING: SECURE_1PSIDTS not set (optional)")
    print()
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['fastapi', 'uvicorn', 'gemini-webapi']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'gemini-webapi':
                __import__('gemini_webapi')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("Install with:")
        print("   pip install -r requirements.txt")
        print()
        return False
    
    print("All dependencies installed")
    print()
    return True

def start_server():
    """Start the API server"""
    print("Starting Gemini API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        import uvicorn
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if check_environment() and check_dependencies():
        start_server()
    else:
        print("Fix the issues above and try again.")
        sys.exit(1)
