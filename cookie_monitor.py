#!/usr/bin/env python3
"""
Cookie health monitoring script
Use this to check if your cookies are valid and when they expire
"""

import asyncio
import os
from datetime import datetime
from gemini_webapi import GeminiClient

async def test_cookies():
    """Test if cookies are working"""
    print("=== Gemini Cookie Health Check ===\n")
    
    # Set your cookies here or load from environment
    secure_1psid = "g.a0003ghjUxBx1PBzZiPo5g2zzwwFUyxGN3gtsu8hXZzQTMTRMyuSb6jFyAyZq_lrkO_pLBus-gACgYKAW8SARQSFQHGX2Mi4M56oRV31xknPEO1q1NrZxoVAUF8yKrc1oWJHd-GreuXGn54GUg30076"
    secure_1psidts = "sidts-CjEBwQ9iI5gH7XWWJzf4b434kSUbkQl2UJ7ZmEPDr93TJNpUwqEAKVI680C3sK2BWR8AEAA"
    
    print(f"Testing cookies...")
    print(f"1PSID length: {len(secure_1psid)}")
    print(f"1PSIDTS length: {len(secure_1psidts)}")
    print()
    
    try:
        client = GeminiClient(secure_1psid, secure_1psidts)
        await client.init(auto_refresh=False)
        print("✅ Cookies are valid and working!")
        
        # Try a simple test request
        response = await client.generate_content(
            "Just say 'OK' - this is a cookie test.",
            model="gemini-2.5-flash"  
        )
        print(f"✅ API test passed - Response: {response.text}")
        
        print("\n📊 Cookie Status: HEALTHY")
        print("📅 Last checked:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ Cookie test failed: {e}")
        print("\n📊 Cookie Status: NEEDS UPDATE")
        print("❗ Action required: Get fresh cookies from gemini.google.com")
        return False

async def check_railway_format():
    """Show cookies in Railway format for easy copy-paste"""
    print("\n=== Railway Variables Format ===\n")
    
    print("Copy these to Railway → Settings → Variables:\n")
    print("SECURE_1PSID=g.a0003ghjUxBx1PBzZiPo5g2zzwwFUyxGN3gtsu8hXZzQTMTRMyuSb6jFyAyZq_lrkO_pLBus-gACgYKAW8SARQSFQHGX2Mi4M56oRV31xknPEO1q1NrZxoVAUF8yKrc1oWJHd-GreuXGn54GUg30076")
    print("SECURE_1PSIDTS=sidts-CjEBwQ9iI5gH7XWWJzf4b434kSUbkQl2UJ7ZmEPDr93TJNpUwqEAKVI680C3sK2BWR8AEAA")
    print()

if __name__ == "__main__":
    print("Gemini Cookie Health Monitor")
    asyncio.run(test_cookies())
    asyncio.run(check_railway_format())
