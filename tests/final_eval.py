# Path: QAgents-workflos/tests/final_eval.py
# Final evaluation - NAKED vs BLACKBOARD on all difficulties
"""Final mode evaluation: NAKED vs fixed BLACKBOARD."""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

api_key = "$env:GOOGLE_API_KEY"
os.environ['GOOGLE_API_KEY'] = api_key

from tests.test_problems import ALL_PROBLEMS
from orchestrators import create_orchestrator
from config import set_api_key
import re

set_api_key(api_key)


def extract_gates(qasm):
    if not qasm:
        return 0
    gate_pattern = r'\b(h|x|y|z|s|t|cx|cz|swap|ccx|rz|rx|ry|cp)\b'
    return len(re.findall(gate_pattern, qasm, re.IGNORECASE))


def test_problem(problem, mode):
    start = time.perf_counter()
    
    try:
        orch = create_orchestrator(mode)
        result = orch.run(problem.prompt)
        
        llm = 1 if mode == "naked" else len(result.agent_results) if result.agent_results else 0
        
        return {
            "success": result.success, 
            "time_ms": (time.perf_counter()-start)*1000,
            "llm": llm, 
            "gates": extract_gates(result.final_output),
            "error": "; ".join(result.errors[:2]) if result.errors else None
        }
            
    except Exception as e:
        return {
            "success": False, 
            "time_ms": (time.perf_counter()-start)*1000, 
            "llm": 0, 
            "gates": 0, 
            "error": str(e)[:60]
        }


print("=" * 80)
print("FINAL MODE EVALUATION: NAKED vs BLACKBOARD")
print("=" * 80)
print(f"Date: {datetime.now().isoformat()}")
print(f"Problems: {len(ALL_PROBLEMS)}")
print()

modes = ["naked", "blackboard"]
results_by_difficulty = {"easy": {}, "medium": {}, "hard": {}, "very_hard": {}}

for problem in ALL_PROBLEMS:
    diff = problem.difficulty.value
    print(f"\n{diff.upper()}: {problem.name}")
    
    if diff not in results_by_difficulty:
        results_by_difficulty[diff] = {}
    
    for mode in modes:
        print(f"  {mode:12}", end=" ", flush=True)
        result = test_problem(problem, mode)
        
        if mode not in results_by_difficulty[diff]:
            results_by_difficulty[diff][mode] = []
        results_by_difficulty[diff][mode].append(result)
        
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['time_ms']:5.0f}ms LLM:{result['llm']} Gates:{result['gates']}")
        
        if result["error"] and not result["success"]:
            print(f"             ⚠️ {result['error'][:50]}...")
        
        time.sleep(4)

# Summary
print("\n\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for mode in modes:
    print(f"\n{mode.upper()}")
    print("-" * 40)
    
    total_success = 0
    total_problems = 0
    total_time = 0
    total_llm = 0
    
    for diff in ["easy", "medium", "hard", "very_hard"]:
        if diff in results_by_difficulty and mode in results_by_difficulty[diff]:
            results = results_by_difficulty[diff][mode]
            successes = sum(1 for r in results if r["success"])
            total_success += successes
            total_problems += len(results)
            total_time += sum(r["time_ms"] for r in results)
            total_llm += sum(r["llm"] for r in results)
            
            print(f"  {diff:10}: {successes}/{len(results)}")
    
    print(f"\n  TOTAL: {total_success}/{total_problems} ({100*total_success/total_problems:.0f}%)")
    print(f"  Time: {total_time:.0f}ms total ({total_time/total_problems:.0f}ms avg)")
    print(f"  LLM calls: {total_llm}")

print("\n" + "=" * 80)
print("WINNER DETERMINATION")
print("=" * 80)

for diff in ["easy", "medium", "hard", "very_hard"]:
    if diff not in results_by_difficulty:
        continue
        
    print(f"\n{diff.upper()}:")
    for mode in modes:
        if mode in results_by_difficulty[diff]:
            results = results_by_difficulty[diff][mode]
            successes = sum(1 for r in results if r["success"])
            avg_time = sum(r["time_ms"] for r in results) / len(results)
            print(f"  {mode}: {successes}/{len(results)} ({avg_time:.0f}ms avg)")

print("\n" + "=" * 80)
print("DONE")
