"""
Database Module: Storage for logs, results, memory, and context.
Provides both shared and per-agent storage with short/long-term memory.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Session-based, cleared on restart
    LONG_TERM = "long_term"    # Persistent across sessions
    SHARED = "shared"          # Shared between agents (blackboard)
    
@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    value: Any
    agent_id: Optional[str]
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
@dataclass
class LogEntry:
    """A log entry for audit trail."""
    level: str
    message: str
    agent_id: Optional[str]
    workflow_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict = field(default_factory=dict)

@dataclass
class ResultEntry:
    """A result from an evaluation run."""
    run_id: str
    system_mode: str  # blackboard, guided, naked
    problem_id: str
    success: bool
    execution_time_ms: float
    circuit_qasm: Optional[str]
    metrics: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class Database:
    """
    SQLite-based storage for all system data.
    Manages logs, results, and agent memory.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_path / "qagents.db"
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_file) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    agent_id TEXT,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    agent_id TEXT,
                    workflow_id TEXT,
                    timestamp TEXT NOT NULL,
                    data TEXT
                );
                
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    system_mode TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    circuit_qasm TEXT,
                    metrics TEXT,
                    timestamp TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key);
                CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory(agent_id);
                CREATE INDEX IF NOT EXISTS idx_results_mode ON results(system_mode);
                CREATE INDEX IF NOT EXISTS idx_results_problem ON results(problem_id);
            """)
            
    # ===== Memory Operations =====
    
    def store_memory(self, entry: MemoryEntry):
        """Store a memory entry."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """INSERT INTO memory (key, value, agent_id, memory_type, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entry.key, json.dumps(entry.value), entry.agent_id,
                 entry.memory_type.value, entry.timestamp.isoformat(),
                 json.dumps(entry.metadata))
            )
            
    def get_memory(self, key: str, agent_id: Optional[str] = None,
                   memory_type: Optional[MemoryType] = None) -> Optional[Any]:
        """Retrieve a memory value."""
        with sqlite3.connect(self.db_file) as conn:
            query = "SELECT value FROM memory WHERE key = ?"
            params = [key]
            
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type.value)
                
            query += " ORDER BY timestamp DESC LIMIT 1"
            
            result = conn.execute(query, params).fetchone()
            return json.loads(result[0]) if result else None
            
    def get_shared_memory(self, key: str) -> Optional[Any]:
        """Get from shared blackboard memory."""
        return self.get_memory(key, memory_type=MemoryType.SHARED)
        
    def set_shared_memory(self, key: str, value: Any, agent_id: Optional[str] = None):
        """Set shared blackboard memory."""
        entry = MemoryEntry(
            key=key,
            value=value,
            agent_id=agent_id,
            memory_type=MemoryType.SHARED
        )
        self.store_memory(entry)
        
    def clear_short_term_memory(self, agent_id: Optional[str] = None):
        """Clear short-term memory (session reset)."""
        with sqlite3.connect(self.db_file) as conn:
            if agent_id:
                conn.execute(
                    "DELETE FROM memory WHERE memory_type = ? AND agent_id = ?",
                    (MemoryType.SHORT_TERM.value, agent_id)
                )
            else:
                conn.execute(
                    "DELETE FROM memory WHERE memory_type = ?",
                    (MemoryType.SHORT_TERM.value,)
                )
                
    # ===== Logging Operations =====
    
    def log(self, entry: LogEntry):
        """Store a log entry."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """INSERT INTO logs (level, message, agent_id, workflow_id, timestamp, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entry.level, entry.message, entry.agent_id, entry.workflow_id,
                 entry.timestamp.isoformat(), json.dumps(entry.data))
            )
            
    def get_logs(self, agent_id: Optional[str] = None, 
                 workflow_id: Optional[str] = None,
                 limit: int = 100) -> List[Dict]:
        """Retrieve log entries."""
        with sqlite3.connect(self.db_file) as conn:
            query = "SELECT * FROM logs WHERE 1=1"
            params = []
            
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            if workflow_id:
                query += " AND workflow_id = ?"
                params.append(workflow_id)
                
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            rows = conn.execute(query, params).fetchall()
            return [
                {"level": r[1], "message": r[2], "agent_id": r[3],
                 "workflow_id": r[4], "timestamp": r[5], "data": json.loads(r[6] or "{}")}
                for r in rows
            ]
            
    # ===== Results Operations =====
    
    def store_result(self, entry: ResultEntry):
        """Store an evaluation result."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """INSERT INTO results (run_id, system_mode, problem_id, success, 
                   execution_time_ms, circuit_qasm, metrics, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry.run_id, entry.system_mode, entry.problem_id,
                 1 if entry.success else 0, entry.execution_time_ms,
                 entry.circuit_qasm, json.dumps(entry.metrics),
                 entry.timestamp.isoformat())
            )
            
    def get_results(self, system_mode: Optional[str] = None,
                    problem_id: Optional[str] = None) -> List[ResultEntry]:
        """Retrieve results for analysis."""
        with sqlite3.connect(self.db_file) as conn:
            query = "SELECT * FROM results WHERE 1=1"
            params = []
            
            if system_mode:
                query += " AND system_mode = ?"
                params.append(system_mode)
            if problem_id:
                query += " AND problem_id = ?"
                params.append(problem_id)
                
            query += " ORDER BY timestamp DESC"
            
            rows = conn.execute(query, params).fetchall()
            return [
                ResultEntry(
                    run_id=r[1], system_mode=r[2], problem_id=r[3],
                    success=bool(r[4]), execution_time_ms=r[5],
                    circuit_qasm=r[6], metrics=json.loads(r[7] or "{}"),
                    timestamp=datetime.fromisoformat(r[8])
                )
                for r in rows
            ]
            
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all runs."""
        with sqlite3.connect(self.db_file) as conn:
            stats = {}
            for mode in ["blackboard", "guided", "naked"]:
                rows = conn.execute(
                    """SELECT COUNT(*), AVG(execution_time_ms), 
                       SUM(success) * 100.0 / COUNT(*)
                       FROM results WHERE system_mode = ?""",
                    (mode,)
                ).fetchone()
                
                stats[mode] = {
                    "total_runs": rows[0] or 0,
                    "avg_time_ms": rows[1] or 0,
                    "success_rate": rows[2] or 0
                }
            return stats


# Singleton instance
_db: Optional[Database] = None

def get_database(db_path: Optional[Path] = None) -> Database:
    """Get or create the database singleton."""
    global _db
    if _db is None:
        from config import config
        path = db_path or config.database.db_path
        _db = Database(path)
    return _db
