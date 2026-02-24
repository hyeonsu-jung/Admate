import time
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class MonitorService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitorService, cls).__new__(cls)
            cls._instance.stats = []
        return cls._instance
    
    def log_request(self, query: str, response_time: float, token_usage: int = 0, source_count: int = 0):
        """요청 통계 기록"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_time": round(response_time, 2),
            "token_usage": token_usage,
            "source_count": source_count
        }
        self.stats.append(entry)
        logger.info(f"MONITOR: Query '{query[:20]}...' handled in {response_time}s")
        
    def get_stats(self) -> List[Dict[str, Any]]:
        """전체 통계 반환"""
        return self.stats

    def get_summary(self) -> Dict[str, Any]:
        """요약 통계 계산"""
        if not self.stats:
            return {"total_queries": 0, "avg_response_time": 0}
            
        total_time = sum(s["response_time"] for s in self.stats)
        return {
            "total_queries": len(self.stats),
            "avg_response_time": round(total_time / len(self.stats), 2),
            "last_request": self.stats[-1]["timestamp"] if self.stats else None
        }

def get_monitor_service():
    return MonitorService()
