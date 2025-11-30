# Path: QAgents-workflos/tests/mode_evaluation.py
# Evaluate all modes on representative problems from each difficulty
"""Mode Evaluation: Test all modes on key problems from each difficulty level."""

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
    PROBLEM_E1_PHASE_FLIP, PROBLEM_E2_CONTROLLED_NOT,
    PROBLEM_M1_SWAP_DECOMPOSITION, PROBLEM_M2_CONTROLLED_Z,
    PROBLEM_H1_DEUTSCH, PROBLEM_H2_GROVER_2QUBIT,
    PROBLEM_VH1_QFT_4QUBIT, PROBLEM_VH2_GROVER_3QUBIT, PROBLEM_VH4_BERNSTEIN_VAZIRANI
)
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
            
        else:
            orch = create_orchestrator(mode)
            result = orch.run(problem.prompt)
            success = result.success
            qasm = result.final_output
            llm = 1 if mode == "naked" else len(result.agent_results) if result.agent_results else 0
            iterations = 1
            
        elapsed = (time.perf_counter() - start) * 1000
        gates = extract_gates(qasm)
        
        return {
            "success": success, 
            "time_ms": elapsed, 
            "llm": llm, 
            "gates": gates,
            "iterations": iterations,
            "error": None
        }
        
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "success": False, 
            "time_ms": elapsed, 
            "llm": 0, 
            "gates": 0,
            "error": str(e)[:80]
        }


def main():
    print("=" * 80)
    print("MODE EVALUATION - KEY PROBLEMS FROM EACH DIFFICULTY")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print()
    
    # Key problems to test (2 per difficulty)
    test_problems = [
        ("EASY", [PROBLEM_E1_PHASE_FLIP, PROBLEM_E2_CONTROLLED_NOT]),
        ("MEDIUM", [PROBLEM_M1_SWAP_DECOMPOSITION, PROBLEM_M2_CONTROLLED_Z]),
        ("HARD", [PROBLEM_H1_DEUTSCH, PROBLEM_H2_GROVER_2QUBIT]),
        ("VERY_HARD", [PROBLEM_VH1_QFT_4QUBIT, PROBLEM_VH2_GROVER_3QUBIT, PROBLEM_VH4_BERNSTEIN_VAZIRANI])
    ]
    
    # Modes to test - focus on working ones
    modes = ["naked", "quasar", "hybrid", "blackboard"]
    
    all_results = []
    
    for diff_name, problems in test_problems:
        print(f"\n{'='*80}")
        print(f"{diff_name} PROBLEMS")
        print("=" * 80)
        
        for problem in problems:
            print(f"\n  {problem.id}: {problem.name}")
            
            for mode in modes:
                print(f"    {mode:12}", end=" ", flush=True)
                
                result = test_problem(problem, mode)
                result["problem_id"] = problem.id
                result["difficulty"] = diff_name.lower()
                result["mode"] = mode
                all_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                time_str = f"{result['time_ms']:6.0f}ms"
                llm_str = f"LLM:{result['llm']}"
                gates_str = f"Gates:{result['gates']:2}"
                
                print(f"{status} {time_str} {llm_str:6} {gates_str}")
                
                if result["error"]:
                    print(f"           ⚠️ {result['error'][:50]}...")
                
                time.sleep(5)  # Rate limiting
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY BY MODE")
    print("=" * 80)
    
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        successes = sum(1 for r in mode_results if r["success"])
        total = len(mode_results)
        total_time = sum(r["time_ms"] for r in mode_results)
        total_llm = sum(r["llm"] for r in mode_results)
        avg_gates = sum(r["gates"] for r in mode_results if r["success"]) / max(successes, 1)
        
        print(f"\n{mode.upper():12}")
        print(f"  Success: {successes}/{total} ({100*successes/total:.0f}%)")
        print(f"  Time: {total_time:.0f}ms total, {total_time/total:.0f}ms avg")
        print(f"  LLM: {total_llm} calls")
        print(f"  Gates: {avg_gates:.1f} avg")
        
        # By difficulty
        for diff in ["easy", "medium", "hard", "very_hard"]:
            diff_results = [r for r in mode_results if r["difficulty"] == diff]
            if diff_results:
                diff_success = sum(1 for r in diff_results if r["success"])
                print(f"    {diff:10}: {diff_success}/{len(diff_results)}")
    
    # Winner by difficulty
    print("\n" + "=" * 80)
    print("🏆 WINNER BY DIFFICULTY")
    print("=" * 80)
    
    for diff in ["easy", "medium", "hard", "very_hard"]:
        diff_results = [r for r in all_results if r["difficulty"] == diff]
        
        print(f"\n{diff.upper()}:")
        for mode in modes:
            mode_diff_results = [r for r in diff_results if r["mode"] == mode]
            if mode_diff_results:
                successes = sum(1 for r in mode_diff_results if r["success"])
                total_time = sum(r["time_ms"] for r in mode_diff_results)
                avg_time = total_time / len(mode_diff_results)
                print(f"  {mode:12} {successes}/{len(mode_diff_results)} ({avg_time:.0f}ms avg)")
    
    # Save results
    output_path = Path(__file__).parent.parent / "research" / f"mode_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
