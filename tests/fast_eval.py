# Path: QAgents-workflos/tests/fast_eval.py
# Fast evaluation - one problem per difficulty, all modes
"""Fast mode evaluation."""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

api_key = "$env:GOOGLE_API_KEY"
os.environ['GOOGLE_API_KEY'] = api_key

from tests.test_problems import (
    PROBLEM_E1_PHASE_FLIP,
    PROBLEM_M1_SWAP_DECOMPOSITION,
    PROBLEM_H1_DEUTSCH,
    PROBLEM_VH4_BERNSTEIN_VAZIRANI
)
from orchestrators import create_orchestrator
from orchestrators.quasar_orchestrator import QuasarOrchestrator, HybridOrchestrator
from config import set_api_key
import re

set_api_key(api_key)


def extract_gates(qasm):
    if not qasm:
        return 0
    gate_pattern = r'\b(h|x|y|z|s|t|cx|cz|swap|ccx|rz|rx|ry|cp)\b'
    return len(re.findall(gate_pattern, qasm, re.IGNORECASE))


def test_problem(problem, mode, timeout=60):
    start = time.perf_counter()
    
    try:
        if mode == "quasar":
            orch = QuasarOrchestrator(max_iterations=3)
            result = orch.run(problem.prompt, problem.expected.min_qubits)
            return {"success": result.success, "time_ms": (time.perf_counter()-start)*1000, 
                    "llm": result.llm_calls, "gates": extract_gates(result.final_qasm), "error": None}
            
        elif mode == "hybrid":
            orch = HybridOrchestrator()
            result = orch.run(problem.prompt, problem.expected.min_qubits)
            return {"success": result.success, "time_ms": (time.perf_counter()-start)*1000,
                    "llm": result.llm_calls, "gates": extract_gates(result.final_qasm), "error": None}
            
        else:
            orch = create_orchestrator(mode)
            result = orch.run(problem.prompt)
            llm = 1 if mode == "naked" else len(result.agent_results) if result.agent_results else 0
            return {"success": result.success, "time_ms": (time.perf_counter()-start)*1000,
                    "llm": llm, "gates": extract_gates(result.final_output), "error": "; ".join(result.errors) if result.errors else None}
            
    except Exception as e:
        return {"success": False, "time_ms": (time.perf_counter()-start)*1000, 
                "llm": 0, "gates": 0, "error": str(e)[:60]}


print("=" * 70)
print("FAST MODE EVALUATION")
print("=" * 70)
print(f"Date: {datetime.now().isoformat()}")

problems = [
    ("EASY", PROBLEM_E1_PHASE_FLIP),
    ("MEDIUM", PROBLEM_M1_SWAP_DECOMPOSITION),
    ("HARD", PROBLEM_H1_DEUTSCH),
    ("VERY_HARD", PROBLEM_VH4_BERNSTEIN_VAZIRANI)
]

modes = ["naked", "quasar", "hybrid", "blackboard"]
all_results = {}

for diff, problem in problems:
    print(f"\n{diff}: {problem.name}")
    print("-" * 50)
    all_results[diff] = {}
    
    for mode in modes:
        print(f"  {mode:12}", end=" ", flush=True)
        result = test_problem(problem, mode)
        all_results[diff][mode] = result
        
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['time_ms']:5.0f}ms LLM:{result['llm']} Gates:{result['gates']}")
        
        if result["error"]:
            print(f"             ⚠️ {result['error'][:40]}...")
        
        time.sleep(5)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for mode in modes:
    successes = sum(1 for diff in all_results if all_results[diff][mode]["success"])
    total_time = sum(all_results[diff][mode]["time_ms"] for diff in all_results)
    total_llm = sum(all_results[diff][mode]["llm"] for diff in all_results)
    print(f"\n{mode.upper():12} {successes}/4 ({25*successes}%) | {total_time:.0f}ms | {total_llm} LLM calls")
    for diff in all_results:
        r = all_results[diff][mode]
        status = "✅" if r["success"] else "❌"
        print(f"  {diff:10} {status}")

print("\n" + "=" * 70)
print("DONE")
