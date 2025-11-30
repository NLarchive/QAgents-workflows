# Path: QAgents-workflos/tests/full_comparison.py
# Full comparison test across all modes and difficulties
"""Full mode comparison test."""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

api_key = "$env:GOOGLE_API_KEY"
os.environ['GOOGLE_API_KEY'] = api_key

from tests.test_problems import ALL_PROBLEMS, ProblemDifficulty
from orchestrators import create_orchestrator
from orchestrators.quasar_orchestrator import QuasarOrchestrator, HybridOrchestrator
from config import set_api_key
import re

set_api_key(api_key)


def extract_gates(qasm):
    """Count gates in QASM."""
    if not qasm:
        return 0
    gate_pattern = r'\b(h|x|y|z|s|t|cx|cz|swap|ccx|rz|rx|ry|cp)\b'
    return len(re.findall(gate_pattern, qasm, re.IGNORECASE))


def test_problem(problem, mode):
    """Test a single problem."""
    start = time.perf_counter()
    
    try:
        if mode == "quasar":
            orch = QuasarOrchestrator(max_iterations=3)
            result = orch.run(
                problem.prompt, 
                problem.expected.min_qubits,
                problem.expected.expected_states if problem.expected.expected_states else None
            )
            success = result.success
            qasm = result.final_qasm
            llm = result.llm_calls
            iterations = result.iterations
            tiers = result.tiers_passed
            
        elif mode == "hybrid":
            orch = HybridOrchestrator()
            result = orch.run(
                problem.prompt, 
                problem.expected.min_qubits,
                problem.expected.expected_states if problem.expected.expected_states else None
            )
            success = result.success
            qasm = result.final_qasm
            llm = result.llm_calls
            iterations = result.iterations
            tiers = result.tiers_passed
            
        else:
            orch = create_orchestrator(mode)
            result = orch.run(problem.prompt)
            success = result.success
            qasm = result.final_output
            llm = 1 if mode == "naked" else len(result.agent_results) if result.agent_results else 0
            iterations = 1
            tiers = []
            
        elapsed = (time.perf_counter() - start) * 1000
        gates = extract_gates(qasm)
        
        return {
            "success": success, 
            "time_ms": elapsed, 
            "llm": llm, 
            "gates": gates,
            "iterations": iterations,
            "tiers": tiers,
            "qasm": qasm,
            "error": None
        }
        
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "success": False, 
            "time_ms": elapsed, 
            "llm": 0, 
            "gates": 0,
            "iterations": 0,
            "tiers": [],
            "qasm": None,
            "error": str(e)[:100]
        }


def main():
    print("=" * 100)
    print("FULL MODE COMPARISON TEST")
    print("=" * 100)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Total problems: {len(ALL_PROBLEMS)}")
    print()
    
    # Modes to test - focus on the key ones
    modes = ["naked", "quasar", "hybrid", "blackboard"]
    
    all_results = []
    
    # Group by difficulty
    for difficulty in [ProblemDifficulty.EASY, ProblemDifficulty.MEDIUM, ProblemDifficulty.HARD, ProblemDifficulty.VERY_HARD]:
        problems = [p for p in ALL_PROBLEMS if p.difficulty == difficulty]
        
        print(f"\n{'='*100}")
        print(f"DIFFICULTY: {difficulty.value.upper()} ({len(problems)} problems)")
        print("=" * 100)
        
        for problem in problems:
            print(f"\n  {problem.id}: {problem.name}")
            
            for mode in modes:
                print(f"    {mode:12}", end=" ", flush=True)
                
                result = test_problem(problem, mode)
                result["problem_id"] = problem.id
                result["difficulty"] = difficulty.value
                result["mode"] = mode
                all_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                time_str = f"{result['time_ms']:6.0f}ms"
                llm_str = f"LLM:{result['llm']}"
                gates_str = f"Gates:{result['gates']:2}"
                
                extra = ""
                if result["tiers"]:
                    extra = f" Tiers:{result['tiers']}"
                
                print(f"{status} {time_str} {llm_str:6} {gates_str}{extra}")
                
                if result["error"]:
                    print(f"           ❌ Error: {result['error'][:60]}...")
                
                time.sleep(5)
    
    # Summary
    print("\n\n" + "=" * 100)
    print("SUMMARY BY MODE")
    print("=" * 100)
    
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        successes = sum(1 for r in mode_results if r["success"])
        total = len(mode_results)
        total_time = sum(r["time_ms"] for r in mode_results)
        total_llm = sum(r["llm"] for r in mode_results)
        avg_gates = sum(r["gates"] for r in mode_results if r["success"]) / max(successes, 1)
        
        print(f"\n{mode.upper():12}")
        print(f"  Overall: {successes}/{total} ({100*successes/total:.0f}%)")
        print(f"  Time: {total_time/1000:.1f}s total, {total_time/total:.0f}ms avg")
        print(f"  LLM: {total_llm} calls ({total_llm/total:.1f} avg)")
        print(f"  Gates: {avg_gates:.1f} avg")
        
        # By difficulty
        for diff in ["easy", "medium", "hard", "very_hard"]:
            diff_results = [r for r in mode_results if r["difficulty"] == diff]
            if diff_results:
                diff_success = sum(1 for r in diff_results if r["success"])
                print(f"    {diff:10}: {diff_success}/{len(diff_results)}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "research" / f"full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean QASM for JSON (can be long)
    for r in all_results:
        if r["qasm"]:
            r["qasm"] = r["qasm"][:500]  # Truncate for storage
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Winner determination
    print("\n" + "=" * 100)
    print("🏆 WINNER BY DIFFICULTY")
    print("=" * 100)
    
    for diff in ["easy", "medium", "hard", "very_hard"]:
        print(f"\n{diff.upper()}:")
        best_mode = None
        best_success = -1
        
        for mode in modes:
            mode_results = [r for r in all_results if r["mode"] == mode and r["difficulty"] == diff]
            if mode_results:
                successes = sum(1 for r in mode_results if r["success"])
                if successes > best_success:
                    best_success = successes
                    best_mode = mode
        
        if best_mode:
            print(f"  🏆 {best_mode.upper()} ({best_success}/{len([r for r in all_results if r['difficulty']==diff and r['mode']==best_mode])})")


if __name__ == "__main__":
    main()
