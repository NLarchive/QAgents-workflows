# Path: QAgents-workflos/tests/mini_test.py
# Description: Test all 4 modes on problems of each difficulty
"""
Mini Test: Comparison of NAKED, BLACKBOARD, GUIDED, HYBRID on 4 problems.
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress Gemini function_call warning (it's informational, not an error)
warnings.filterwarnings("ignore", message=".*non-text parts.*")

from orchestrators import create_orchestrator
from tests.test_problems import get_problems_by_difficulty, ProblemDifficulty as Difficulty

def test_one(problem, mode):
    """Test a single problem with a mode."""
    orch = create_orchestrator(mode)
    import time
    start = time.perf_counter()
    result = orch.run(problem.prompt)
    elapsed = (time.perf_counter() - start) * 1000
    
    # Count gates
    gates = 0
    if result.final_output:
        gates = len([l for l in result.final_output.split('\n') 
                    if l.strip() and not l.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'measure', '//'))])
    
    return result.success, elapsed, gates

def main():
    print("=" * 70)
    print("COMPREHENSIVE TEST: NAKED vs BLACKBOARD vs GUIDED vs HYBRID")
    print("=" * 70)
    
    # Test HARD problems to see where modes fail
    modes = ["naked", "blackboard", "guided", "hybrid"]
    
    # One problem per difficulty
    test_problems = [
        ("EASY", get_problems_by_difficulty(Difficulty.EASY)[0]),
        ("HARD", get_problems_by_difficulty(Difficulty.HARD)[0]),
        ("VERY_HARD", get_problems_by_difficulty(Difficulty.VERY_HARD)[0]),
    ]
    
    results = {mode: [] for mode in modes}
    
    for diff_name, problem in test_problems:
        print(f"\n{diff_name}: {problem.name}")
        print("-" * 50)
        
        for mode in modes:
            try:
                ok, ms, gates = test_one(problem, mode)
                status = "✅" if ok else "❌"
                print(f"  {mode:12} {status} {ms:6.0f}ms {gates:2} gates")
                results[mode].append(ok)
            except Exception as e:
                print(f"  {mode:12} ❌ Error: {str(e)[:50]}")
                results[mode].append(False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mode in modes:
        passed = sum(results[mode])
        total = len(results[mode])
        pct = 100*passed/total if total > 0 else 0
        print(f"  {mode:12}: {passed}/{total} passed ({pct:.0f}%)")

if __name__ == "__main__":
    main()
