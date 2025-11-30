# Path: QAgents-workflos/tests/quick_mode_test.py
# Description: Quick test of all modes on one HARD problem
"""
Quick Mode Test: Test all 4 modes on 1 problem each difficulty
Designed to be fast by testing only essential combinations.
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings
warnings.filterwarnings("ignore", message=".*non-text parts.*")
warnings.filterwarnings("ignore", message=".*GOOGLE_API_KEY.*")

import time
from orchestrators import create_orchestrator
from tests.test_problems import get_problems_by_difficulty, ProblemDifficulty

def test_mode(mode, problem):
    """Test a single mode on a problem."""
    try:
        orch = create_orchestrator(mode)
        start = time.perf_counter()
        result = orch.run(problem.prompt)
        elapsed = (time.perf_counter() - start) * 1000
        
        gates = 0
        if result.final_output:
            gates = len([l for l in result.final_output.split('\n') 
                        if l.strip() and not l.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'measure', '//'))])
        
        return result.success, elapsed, gates, None
    except Exception as e:
        return False, 0, 0, str(e)[:50]

def main():
    print("=" * 60)
    print("QUICK MODE TEST: All 4 modes on HARD problem")
    print("=" * 60)
    
    # Get one VERY_HARD problem - this will show where modes struggle
    very_hard_problems = get_problems_by_difficulty(ProblemDifficulty.VERY_HARD)
    problem = very_hard_problems[0]  # 4-Qubit QFT
    
    print(f"\nProblem: {problem.name}")
    print(f"Difficulty: VERY_HARD")
    print(f"Description: {problem.prompt[:80]}...")
    print("-" * 60)
    
    modes = ["naked", "quasar", "hybrid", "blackboard"]
    results = []
    
    for mode in modes:
        print(f"\nTesting {mode}...", end=" ", flush=True)
        ok, ms, gates, error = test_mode(mode, problem)
        
        if ok:
            print(f"✅ {ms:.0f}ms, {gates} gates")
            results.append((mode, True, ms, gates))
        elif error:
            print(f"❌ Error: {error}")
            results.append((mode, False, 0, 0))
        else:
            print(f"❌ Failed ({ms:.0f}ms)")
            results.append((mode, False, ms, gates))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for mode, ok, ms, gates in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {mode:12}: {status:10} {ms:6.0f}ms  {gates:2} gates")
    
    passed = sum(1 for r in results if r[1])
    print(f"\nTotal: {passed}/{len(results)} modes passed")

if __name__ == "__main__":
    main()
