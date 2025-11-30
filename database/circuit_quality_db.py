# Path: QAgents-workflos/database/circuit_quality_db.py
# Relations: Uses database/storage.py pattern, connects to MCP via client/
# Description: SQLite database for storing QASM circuits and quality metrics
#              Enables circuit comparison across orchestration modes
#              Tracks circuit_qasm text + all quality measurements

"""
Circuit Quality Database: Store and compare quantum circuits with quality metrics.
Stores actual QASM code for later analysis and comparison between modes.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a circuit."""
    depth: int = 0
    gate_count: int = 0
    cx_count: int = 0
    single_qubit_count: int = 0
    hardware_fitness: float = 0.0
    syntax_valid: bool = False
    state_correctness: float = 0.0
    complexity_score: float = 0.0
    noise_estimate: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate overall quality score (higher is better, 0-100)."""
        score = 0.0
        # Syntax: 20 points
        score += 20.0 if self.syntax_valid else 0.0
        # Hardware fitness: 20 points
        score += 20.0 * min(self.hardware_fitness, 1.0)
        # State correctness: 30 points
        score += 30.0 * self.state_correctness
        # Efficiency (lower depth/gates better): 15 points
        if self.gate_count > 0:
            efficiency = max(0, 1 - (self.depth / max(self.gate_count, 1)) / 10)
            score += 15.0 * efficiency
        # Lower CX count bonus: 15 points
        if self.gate_count > 0:
            cx_ratio = self.cx_count / max(self.gate_count, 1)
            score += 15.0 * (1 - min(cx_ratio, 1.0))
        return round(score, 2)


@dataclass
class CircuitEvaluation:
    """Complete evaluation record with QASM and quality."""
    id: Optional[int] = None
    run_id: str = ""
    timestamp: str = ""
    problem_id: str = ""
    problem_goal: str = ""
    mode: str = ""  # naked, guided, blackboard
    qasm_code: str = ""  # FULL QASM text stored
    success: bool = False
    execution_time_ms: float = 0.0
    llm_requests: int = 0
    tokens_used: int = 0
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    errors: List[str] = field(default_factory=list)


class CircuitQualityDB:
    """
    SQLite database for storing circuits and quality metrics.
    Primary purpose: Enable quality comparison across modes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "data"
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_path / "circuit_quality.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_file) as conn:
            conn.executescript("""
                -- Main table: stores full QASM and evaluation metadata
                CREATE TABLE IF NOT EXISTS circuit_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    problem_goal TEXT,
                    mode TEXT NOT NULL,
                    qasm_code TEXT,
                    success INTEGER NOT NULL,
                    execution_time_ms REAL,
                    llm_requests INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    errors TEXT
                );
                
                -- Quality metrics table: detailed quality measurements
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    eval_id INTEGER NOT NULL,
                    depth INTEGER DEFAULT 0,
                    gate_count INTEGER DEFAULT 0,
                    cx_count INTEGER DEFAULT 0,
                    single_qubit_count INTEGER DEFAULT 0,
                    hardware_fitness REAL DEFAULT 0.0,
                    syntax_valid INTEGER DEFAULT 0,
                    state_correctness REAL DEFAULT 0.0,
                    complexity_score REAL DEFAULT 0.0,
                    noise_estimate REAL DEFAULT 0.0,
                    overall_score REAL DEFAULT 0.0,
                    FOREIGN KEY (eval_id) REFERENCES circuit_evaluations(id)
                );
                
                -- Comparison runs: group multiple evaluations
                CREATE TABLE IF NOT EXISTS comparison_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT,
                    num_problems INTEGER DEFAULT 0,
                    modes_tested TEXT,
                    summary TEXT
                );
                
                -- Create indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_eval_run_id ON circuit_evaluations(run_id);
                CREATE INDEX IF NOT EXISTS idx_eval_problem ON circuit_evaluations(problem_id);
                CREATE INDEX IF NOT EXISTS idx_eval_mode ON circuit_evaluations(mode);
            """)
            conn.commit()
    
    def save_evaluation(self, eval: CircuitEvaluation) -> int:
        """Save a circuit evaluation with quality metrics. Returns eval ID."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Insert main evaluation record
            cursor.execute("""
                INSERT INTO circuit_evaluations 
                (run_id, timestamp, problem_id, problem_goal, mode, qasm_code,
                 success, execution_time_ms, llm_requests, tokens_used, errors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval.run_id,
                eval.timestamp or datetime.now().isoformat(),
                eval.problem_id,
                eval.problem_goal,
                eval.mode,
                eval.qasm_code,  # FULL QASM stored here
                1 if eval.success else 0,
                eval.execution_time_ms,
                eval.llm_requests,
                eval.tokens_used,
                json.dumps(eval.errors)
            ))
            eval_id = cursor.lastrowid
            
            # Insert quality metrics
            metrics = eval.quality_metrics
            cursor.execute("""
                INSERT INTO quality_metrics
                (eval_id, depth, gate_count, cx_count, single_qubit_count,
                 hardware_fitness, syntax_valid, state_correctness,
                 complexity_score, noise_estimate, overall_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_id,
                metrics.depth,
                metrics.gate_count,
                metrics.cx_count,
                metrics.single_qubit_count,
                metrics.hardware_fitness,
                1 if metrics.syntax_valid else 0,
                metrics.state_correctness,
                metrics.complexity_score,
                metrics.noise_estimate,
                metrics.overall_score()
            ))
            
            conn.commit()
            logger.info(f"Saved evaluation {eval_id} for {eval.problem_id}/{eval.mode}")
            return eval_id
    
    def save_comparison_run(self, run_id: str, description: str, 
                           num_problems: int, modes: List[str], summary: Dict) -> None:
        """Save a comparison run record."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO comparison_runs
                (run_id, timestamp, description, num_problems, modes_tested, summary)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                datetime.now().isoformat(),
                description,
                num_problems,
                json.dumps(modes),
                json.dumps(summary)
            ))
            conn.commit()
    
    def get_evaluations(self, problem_id: Optional[str] = None, 
                       mode: Optional[str] = None,
                       run_id: Optional[str] = None,
                       limit: int = 100) -> List[CircuitEvaluation]:
        """Get evaluations with optional filters."""
        query = """
            SELECT e.*, q.depth, q.gate_count, q.cx_count, q.single_qubit_count,
                   q.hardware_fitness, q.syntax_valid, q.state_correctness,
                   q.complexity_score, q.noise_estimate, q.overall_score
            FROM circuit_evaluations e
            LEFT JOIN quality_metrics q ON e.id = q.eval_id
            WHERE 1=1
        """
        params = []
        
        if problem_id:
            query += " AND e.problem_id = ?"
            params.append(problem_id)
        if mode:
            query += " AND e.mode = ?"
            params.append(mode)
        if run_id:
            query += " AND e.run_id = ?"
            params.append(run_id)
        
        query += " ORDER BY e.timestamp DESC LIMIT ?"
        params.append(limit)
        
        evaluations = []
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor:
                metrics = QualityMetrics(
                    depth=row['depth'] or 0,
                    gate_count=row['gate_count'] or 0,
                    cx_count=row['cx_count'] or 0,
                    single_qubit_count=row['single_qubit_count'] or 0,
                    hardware_fitness=row['hardware_fitness'] or 0.0,
                    syntax_valid=bool(row['syntax_valid']),
                    state_correctness=row['state_correctness'] or 0.0,
                    complexity_score=row['complexity_score'] or 0.0,
                    noise_estimate=row['noise_estimate'] or 0.0
                )
                
                eval = CircuitEvaluation(
                    id=row['id'],
                    run_id=row['run_id'],
                    timestamp=row['timestamp'],
                    problem_id=row['problem_id'],
                    problem_goal=row['problem_goal'] or "",
                    mode=row['mode'],
                    qasm_code=row['qasm_code'] or "",
                    success=bool(row['success']),
                    execution_time_ms=row['execution_time_ms'] or 0.0,
                    llm_requests=row['llm_requests'] or 0,
                    tokens_used=row['tokens_used'] or 0,
                    quality_metrics=metrics,
                    errors=json.loads(row['errors']) if row['errors'] else []
                )
                evaluations.append(eval)
        
        return evaluations
    
    def get_circuit_by_id(self, eval_id: int) -> Optional[CircuitEvaluation]:
        """Get a single evaluation by ID."""
        evals = self.get_evaluations(limit=1)
        for e in self.get_evaluations(limit=1000):
            if e.id == eval_id:
                return e
        return None
    
    def compare_modes_for_problem(self, problem_id: str, run_id: Optional[str] = None) -> Dict:
        """Compare all modes for a specific problem."""
        modes = ['naked', 'guided', 'blackboard']
        comparison = {
            "problem_id": problem_id,
            "modes": {}
        }
        
        for mode in modes:
            evals = self.get_evaluations(problem_id=problem_id, mode=mode, run_id=run_id)
            if evals:
                latest = evals[0]
                comparison["modes"][mode] = {
                    "success": latest.success,
                    "qasm_code": latest.qasm_code,
                    "depth": latest.quality_metrics.depth,
                    "gate_count": latest.quality_metrics.gate_count,
                    "cx_count": latest.quality_metrics.cx_count,
                    "hardware_fitness": latest.quality_metrics.hardware_fitness,
                    "overall_score": latest.quality_metrics.overall_score(),
                    "execution_time_ms": latest.execution_time_ms,
                    "llm_requests": latest.llm_requests
                }
        
        return comparison
    
    def get_quality_summary(self, run_id: Optional[str] = None) -> Dict:
        """Get quality summary across all modes."""
        query = """
            SELECT e.mode, 
                   COUNT(*) as count,
                   SUM(e.success) as successes,
                   AVG(q.overall_score) as avg_score,
                   AVG(q.depth) as avg_depth,
                   AVG(q.gate_count) as avg_gates,
                   AVG(q.cx_count) as avg_cx,
                   AVG(q.hardware_fitness) as avg_fitness,
                   AVG(e.execution_time_ms) as avg_time,
                   SUM(e.llm_requests) as total_llm,
                   SUM(e.tokens_used) as total_tokens
            FROM circuit_evaluations e
            LEFT JOIN quality_metrics q ON e.id = q.eval_id
        """
        params = []
        if run_id:
            query += " WHERE e.run_id = ?"
            params.append(run_id)
        query += " GROUP BY e.mode"
        
        summary = {"modes": {}}
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(query, params):
                mode = row['mode']
                count = row['count']
                summary["modes"][mode] = {
                    "count": count,
                    "success_rate": row['successes'] / count if count > 0 else 0,
                    "avg_quality_score": round(row['avg_score'] or 0, 2),
                    "avg_depth": round(row['avg_depth'] or 0, 1),
                    "avg_gates": round(row['avg_gates'] or 0, 1),
                    "avg_cx_count": round(row['avg_cx'] or 0, 1),
                    "avg_hardware_fitness": round(row['avg_fitness'] or 0, 3),
                    "avg_time_ms": round(row['avg_time'] or 0, 1),
                    "total_llm_requests": row['total_llm'] or 0,
                    "total_tokens": row['total_tokens'] or 0
                }
        
        return summary
    
    def export_circuits_markdown(self, run_id: Optional[str] = None) -> str:
        """Export all circuits as markdown for comparison."""
        evals = self.get_evaluations(run_id=run_id, limit=1000)
        
        # Group by problem
        by_problem: Dict[str, Dict[str, CircuitEvaluation]] = {}
        for e in evals:
            if e.problem_id not in by_problem:
                by_problem[e.problem_id] = {}
            by_problem[e.problem_id][e.mode] = e
        
        md = ["# Circuit Quality Comparison Report\n"]
        md.append(f"Generated: {datetime.now().isoformat()}\n")
        if run_id:
            md.append(f"Run ID: {run_id}\n")
        md.append("\n---\n")
        
        for problem_id, modes in sorted(by_problem.items()):
            md.append(f"\n## Problem: {problem_id}\n")
            
            for mode in ['naked', 'guided', 'blackboard']:
                if mode not in modes:
                    md.append(f"\n### {mode.upper()}: NOT RUN\n")
                    continue
                
                e = modes[mode]
                q = e.quality_metrics
                
                md.append(f"\n### {mode.upper()}\n")
                md.append(f"- **Success**: {'✅' if e.success else '❌'}\n")
                md.append(f"- **Quality Score**: {q.overall_score()}/100\n")
                md.append(f"- **Depth**: {q.depth}\n")
                md.append(f"- **Gate Count**: {q.gate_count}\n")
                md.append(f"- **CX Count**: {q.cx_count}\n")
                md.append(f"- **Hardware Fitness**: {q.hardware_fitness:.3f}\n")
                md.append(f"- **Time**: {e.execution_time_ms:.0f}ms\n")
                md.append(f"- **LLM Requests**: {e.llm_requests}\n")
                
                if e.qasm_code:
                    md.append("\n```qasm\n")
                    md.append(e.qasm_code)
                    if not e.qasm_code.endswith('\n'):
                        md.append('\n')
                    md.append("```\n")
                else:
                    md.append("\n*No circuit generated*\n")
        
        return "".join(md)


# Singleton instance
_quality_db: Optional[CircuitQualityDB] = None

def get_quality_db() -> CircuitQualityDB:
    """Get the global quality database instance."""
    global _quality_db
    if _quality_db is None:
        _quality_db = CircuitQualityDB()
    return _quality_db
