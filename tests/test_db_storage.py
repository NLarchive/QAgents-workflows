# Path: QAgents-workflos/test_db_storage.py
# Description: Quick test to verify database storage works
"""Test that database can store and retrieve circuits."""

from database.circuit_quality_db import CircuitQualityDB, CircuitEvaluation, QualityMetrics, get_quality_db
from datetime import datetime

def test_db():
    # Test database
    db = get_quality_db()
    print(f'Database file: {db.db_file}')

    # Create a test evaluation with sample QASM
    test_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

    test_eval = CircuitEvaluation(
        run_id='test_manual_001',
        timestamp=datetime.now().isoformat(),
        problem_id='test_bell_state',
        problem_goal='Create Bell state',
        mode='manual_test',
        qasm_code=test_qasm,
        success=True,
        execution_time_ms=0,
        llm_requests=0,
        tokens_used=0,
        quality_metrics=QualityMetrics(
            depth=2,
            gate_count=3,
            cx_count=1,
            single_qubit_count=1,
            hardware_fitness=0.95,
            syntax_valid=True,
            state_correctness=1.0
        )
    )

    # Save to database
    eval_id = db.save_evaluation(test_eval)
    print(f'Saved evaluation ID: {eval_id}')

    # Retrieve and verify
    evals = db.get_evaluations(problem_id='test_bell_state')
    print(f'Retrieved {len(evals)} evaluations')
    if evals:
        e = evals[0]
        print(f'QASM stored ({len(e.qasm_code)} chars):')
        print(e.qasm_code)
        print(f'Quality score: {e.quality_metrics.overall_score()}/100')

if __name__ == "__main__":
    test_db()
