#!/usr/bin/env python3
"""
Background scheduler for proactive cookie refresh
This ensures cookies stay fresh even without API requests
"""

import asyncio
import logging
import os
from datetime import datetime
from contextlib import asynccontextmanager

from production_server import clients, get_or_init_client
from gemini_webapi.constants import Model

logger = logging.getLogger(__name__)

class CookieRefreshScheduler:
    """Background task to refresh cookies proactively"""
    
    def __init__(self, refresh_interval_minutes: int = 9):
        self.refresh_interval = refresh_interval_minutes * 60  # Convert to seconds
        self.running = False
        self.task = None
        
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
        
    async def _refresh_loop(self):
        """Main refresh loop that runs periodically"""
        while self.running:
            try:
                await asyncio.sleep(self.refresh_interval)
                if not self.running:
                    break
                    
                await self._refresh_all_clients()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cookie refresh loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    async def _refresh_all_clients(self):
        """Refresh all active Gemini clients"""
        if not clients:
            logger.info("No clients to refresh")
            return
            
        refresh_count = 0
        for account_id, client in list(clients.items()):
            try:
                # Make a lightweight request to trigger auto-refresh
                await client.generate_content(
                    "test",  # Minimal request to trigger refresh
                    model=Model.G_2_5_FLASH
                )
                refresh_count += 1
                logger.info(f"Refreshed cookies for account: {account_id}")
                
            except Exception as e:
                logger.error(f"Failed to refresh cookies for account {account_id}: {e}")
                
        if refresh_count > 0:
            logger.info(f"Successfully refreshed {refresh_count} account(s) at {datetime.utcnow().isoformat()}")
            
# Global scheduler instance
scheduler = CookieRefreshScheduler()
