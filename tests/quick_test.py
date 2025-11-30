# Path: QAgents-workflos/tests/quick_test.py
# Quick test to compare modes on easy problems only
"""Quick test for mode comparison."""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

api_key = "$env:GOOGLE_API_KEY"
os.environ['GOOGLE_API_KEY'] = api_key

from tests.test_problems import EASY_PROBLEMS, VERY_HARD_PROBLEMS
from orchestrators import create_orchestrator
from orchestrators.quasar_orchestrator import QuasarOrchestrator, HybridOrchestrator
from config import set_api_key

set_api_key(api_key)

def test_problem(problem, mode):
    """Test a single problem."""
    start = time.perf_counter()
    
    try:
        if mode == "quasar":
            orch = QuasarOrchestrator(max_iterations=3)
            result = orch.run(problem.prompt, problem.expected.min_qubits)
            success = result.success
            qasm = result.final_qasm
            llm = result.llm_calls
        elif mode == "hybrid":
            orch = HybridOrchestrator()
            result = orch.run(problem.prompt, problem.expected.min_qubits)
            success = result.success
            qasm = result.final_qasm
            llm = result.llm_calls
        else:
            orch = create_orchestrator(mode)
            result = orch.run(problem.prompt)
            success = result.success
            qasm = result.final_output
            llm = len([k for k in result.agent_results.keys()]) if result.agent_results else 1
            
        elapsed = (time.perf_counter() - start) * 1000
        return {"success": success, "time_ms": elapsed, "llm": llm, "qasm": qasm[:100] if qasm else None}
        
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"success": False, "time_ms": elapsed, "llm": 0, "error": str(e)[:50]}

print("=" * 80)
print("QUICK MODE COMPARISON TEST")
print("=" * 80)

# Test only first easy and first very_hard problem with all modes
test_cases = [
    ("EASY", EASY_PROBLEMS[0]),
    ("VERY_HARD", VERY_HARD_PROBLEMS[0])
]

modes = ["naked", "quasar", "hybrid"]  # Skip slow modes

for diff, problem in test_cases:
    print(f"\n{diff}: {problem.name}")
    print("-" * 60)
    
    for mode in modes:
        print(f"  {mode}...", end=" ", flush=True)
        result = test_problem(problem, mode)
        
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time_ms']:.0f}ms"
        llm_str = f"LLM:{result.get('llm', '?')}"
        
        print(f"{status} {time_str} {llm_str}")
        
        if not result["success"] and "error" in result:
            print(f"    Error: {result['error']}")
            
        time.sleep(5)  # Rate limiting

print("\n" + "=" * 80)
print("DONE")
