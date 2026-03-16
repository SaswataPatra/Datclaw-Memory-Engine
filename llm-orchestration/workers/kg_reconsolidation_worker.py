"""
KG Re-consolidation Worker

Background worker that periodically runs KG re-consolidation for all users.
Can be triggered manually or run on a schedule.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class KGReconsolidationWorker:
    """
    Worker that periodically runs KG re-consolidation.
    """

    def __init__(self, reconsolidation_service, arango_db, interval_hours: int = 24):
        self.reconsolidation_service = reconsolidation_service
        self.arango_db = arango_db
        self.interval_hours = interval_hours
        self.running = False

    async def run_for_user(self, user_id: str) -> dict:
        """
        Run re-consolidation for a single user.
        
        Returns:
            Summary of operations
        """
        try:
            result = await self.reconsolidation_service.run_full_reconsolidation(user_id)
            return result
        except Exception as e:
            logger.error(f"Re-consolidation failed for user {user_id}: {e}", exc_info=True)
            return {
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run_for_all_users(self) -> list:
        """
        Run re-consolidation for all users with memories.
        
        Returns:
            List of summaries for each user
        """
        logger.info("🔄 Starting KG re-consolidation for all users")
        
        try:
            # Get all unique user IDs from memories collection
            query = """
            FOR mem IN memories
                COLLECT user_id = mem.user_id
                RETURN user_id
            """
            
            cursor = self.arango_db.aql.execute(query)
            user_ids = list(cursor)
            
            if not user_ids:
                logger.info("   No users found")
                return []
            
            logger.info(f"   Found {len(user_ids)} users")
            
            results = []
            for user_id in user_ids:
                result = await self.run_for_user(user_id)
                results.append(result)
            
            logger.info(f"✅ Re-consolidation complete for {len(user_ids)} users")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run re-consolidation for all users: {e}", exc_info=True)
            return []

    async def start_periodic(self):
        """
        Start periodic re-consolidation (runs every interval_hours).
        """
        self.running = True
        logger.info(f"🔄 Starting periodic KG re-consolidation (interval: {self.interval_hours}h)")
        
        while self.running:
            try:
                await self.run_for_all_users()
            except Exception as e:
                logger.error(f"Periodic re-consolidation failed: {e}", exc_info=True)
            
            # Wait for next interval
            await asyncio.sleep(self.interval_hours * 3600)

    def stop(self):
        """
        Stop periodic re-consolidation.
        """
        self.running = False
        logger.info("🛑 Stopping periodic KG re-consolidation")
