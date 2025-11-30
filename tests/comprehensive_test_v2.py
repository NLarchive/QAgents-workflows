# Path: QAgents-workflos/tests/comprehensive_test_v2.py
# Relations: Uses orchestrators, test_problems, client/mcp_client
# Description: Full diagnostic test comparing all 5 modes including QUASAR and HYBRID
"""
Comprehensive Test V2: Compare all orchestration modes

Modes tested:
1. NAKED - Direct LLM (baseline)
2. GUIDED - Multi-agent pipeline  
3. BLACKBOARD - Event-driven agents
4. QUASAR - Tool-augmented LLM with hierarchical validation
5. HYBRID - NAKED first, QUASAR fallback

Problems:
- 3 EASY
- 3 MEDIUM  
- 3 HARD
- 4 VERY_HARD (new - to find NAKED limits)
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Set API key BEFORE any imports
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    api_key = "$env:GOOGLE_API_KEY"
    os.environ['GOOGLE_API_KEY'] = api_key

from tests.test_problems import (
    ALL_PROBLEMS, EASY_PROBLEMS, MEDIUM_PROBLEMS, 
    HARD_PROBLEMS, VERY_HARD_PROBLEMS,
    ProblemDifficulty
)
from orchestrators import create_orchestrator
from orchestrators.quasar_orchestrator import QuasarOrchestrator, HybridOrchestrator
from config import reset_cost_tracking, get_cost_summary, set_api_key
from client.mcp_client import get_client

# Set API key in config
set_api_key(api_key)


def extract_qasm_metrics(qasm: str) -> dict:
    """Extract metrics from QASM code."""
    if not qasm:
        return {"gate_count": 0, "depth": 0, "qubits": 0}
    
    import re
    
    # Count qubits
    qreg_match = re.search(r'qreg\s+\w+\[(\d+)\]', qasm)
    qubits = int(qreg_match.group(1)) if qreg_match else 0
    
    # Count gates (excluding declarations and measurements)
    gate_pattern = r'\b(h|x|y|z|s|t|sdg|tdg|cx|cz|cy|swap|ccx|rz|rx|ry|u1|u2|u3|p|cp)\b'
    gates = re.findall(gate_pattern, qasm, re.IGNORECASE)
    
    # Estimate depth (simplified)
    lines = [l.strip() for l in qasm.split('\n') if l.strip() and not l.strip().startswith(('OPENQASM', 'include', 'qreg', 'creg', '//'))]
    depth = len([l for l in lines if any(g in l.lower() for g in ['h ', 'x ', 'y ', 'z ', 'cx', 'cz', 'swap', 'rx', 'ry', 'rz', 'ccx'])])
    
    return {"gate_count": len(gates), "depth": depth, "qubits": qubits}


def run_test(problem, mode: str) -> dict:
    """Run a single test and return results."""
    result = {
        "problem_id": problem.id,
        "problem_name": problem.name,
        "difficulty": problem.difficulty.value,
        "category": problem.category.value,
        "mode": mode,
        "success": False,
        "qasm_valid": False,
        "time_ms": 0,
        "llm_calls": 0,
        "tokens": 0,
        "gate_count": 0,
        "depth": 0,
        "qasm": None,
        "error": None,
        "tiers_passed": [],
        "iterations": 0
    }
    
    start = time.perf_counter()
    reset_cost_tracking()
    
    try:
        if mode in ["quasar", "hybrid"]:
            # Use new orchestrators with expected values
            if mode == "quasar":
                orchestrator = QuasarOrchestrator(max_iterations=3)
            else:
                orchestrator = HybridOrchestrator()
            
            quasar_result = orchestrator.run(
                goal=problem.prompt,
                expected_qubits=problem.expected.min_qubits,
                expected_states=problem.expected.expected_states if problem.expected.expected_states else None,
                max_depth=problem.expected.max_depth
            )
            
            result["success"] = quasar_result.success
            result["qasm"] = quasar_result.final_qasm
            result["llm_calls"] = quasar_result.llm_calls
            result["tokens"] = quasar_result.tokens_used
            result["tiers_passed"] = quasar_result.tiers_passed
            result["iterations"] = quasar_result.iterations
            
            if quasar_result.final_qasm:
                result["qasm_valid"] = True
                metrics = extract_qasm_metrics(quasar_result.final_qasm)
                result["gate_count"] = metrics["gate_count"]
                result["depth"] = metrics["depth"]
            
            if quasar_result.errors:
                result["error"] = "; ".join(quasar_result.errors)
                
        else:
            # Use standard orchestrators
            orchestrator = create_orchestrator(mode)
            orch_result = orchestrator.run(problem.prompt)
            
            result["success"] = orch_result.success
            result["qasm"] = orch_result.final_output
            
            # Get LLM stats
            cost = get_cost_summary()
            result["llm_calls"] = cost.get("llm_requests", 0)
            result["tokens"] = cost.get("total_tokens", 0)
            
            if orch_result.final_output:
                result["qasm_valid"] = True
                metrics = extract_qasm_metrics(orch_result.final_output)
                result["gate_count"] = metrics["gate_count"]
                result["depth"] = metrics["depth"]
            
            if orch_result.errors:
                result["error"] = "; ".join(orch_result.errors)
                
    except Exception as e:
        result["error"] = str(e)
        
    result["time_ms"] = (time.perf_counter() - start) * 1000
    return result


def main():
    print("=" * 100)
    print("COMPREHENSIVE TEST V2 - ALL MODES INCLUDING QUASAR & HYBRID")
    print("=" * 100)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Problems: {len(ALL_PROBLEMS)} total")
    print(f"  - Easy: {len(EASY_PROBLEMS)}")
    print(f"  - Medium: {len(MEDIUM_PROBLEMS)}")
    print(f"  - Hard: {len(HARD_PROBLEMS)}")
    print(f"  - Very Hard: {len(VERY_HARD_PROBLEMS)}")
    print(f"Modes: naked, guided, blackboard, quasar, hybrid")
    print("=" * 100)
    
    # Check MCP server
    try:
        client = get_client()
        if client.health_check():
            print("✅ MCP Server connected")
        else:
            print("⚠️ MCP Server not responding - some validations may use fallback")
    except:
        print("⚠️ MCP Server not available")
    
    all_results = []
    modes = ["naked", "quasar", "hybrid", "guided", "blackboard"]  # Order: fastest to slowest
    
    # Group problems by difficulty
    problem_groups = [
        ("EASY", EASY_PROBLEMS),
        ("MEDIUM", MEDIUM_PROBLEMS),
        ("HARD", HARD_PROBLEMS),
        ("VERY_HARD", VERY_HARD_PROBLEMS)
    ]
    
    for diff_name, problems in problem_groups:
        print(f"\n{'='*100}")
        print(f"DIFFICULTY: {diff_name}")
        print("=" * 100)
        
        for problem in problems:
            print(f"\n--- Problem: {problem.id} - {problem.name} ---")
            
            for mode in modes:
                print(f"  Testing {mode}...", end=" ", flush=True)
                
                result = run_test(problem, mode)
                all_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                time_str = f"{result['time_ms']:.0f}ms"
                llm_str = f"LLM:{result['llm_calls']}"
                gates_str = f"Gates:{result['gate_count']}"
                
                extra = ""
                if mode in ["quasar", "hybrid"]:
                    tiers = result.get("tiers_passed", [])
                    extra = f" Tiers:{tiers}"
                
                print(f"{status} {time_str} {llm_str} {gates_str}{extra}")
                
                if result["error"] and not result["success"]:
                    print(f"    Error: {result['error'][:80]}...")
                
                # Rate limiting
                time.sleep(5)
    
    # Summary
    print("\n\n" + "=" * 100)
    print("FINAL SUMMARY BY MODE")
    print("=" * 100)
    
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        successes = sum(1 for r in mode_results if r["success"])
        total = len(mode_results)
        total_time = sum(r["time_ms"] for r in mode_results)
        total_llm = sum(r["llm_calls"] for r in mode_results)
        avg_gates = sum(r["gate_count"] for r in mode_results if r["success"]) / max(successes, 1)
        
        print(f"\n{mode.upper()}:")
        print(f"  Success: {successes}/{total} ({100*successes/total:.1f}%)")
        print(f"  Total Time: {total_time:.0f}ms ({total_time/total:.0f}ms avg)")
        print(f"  LLM Calls: {total_llm} ({total_llm/total:.1f} avg)")
        print(f"  Avg Gates (success): {avg_gates:.1f}")
        
        # Per difficulty
        for diff in ["easy", "medium", "hard", "very_hard"]:
            diff_results = [r for r in mode_results if r["difficulty"] == diff]
            if diff_results:
                diff_success = sum(1 for r in diff_results if r["success"])
                print(f"    {diff}: {diff_success}/{len(diff_results)}")
    
    # Efficiency comparison
    print("\n" + "=" * 100)
    print("EFFICIENCY COMPARISON (Success per LLM call)")
    print("=" * 100)
    
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        successes = sum(1 for r in mode_results if r["success"])
        total_llm = sum(r["llm_calls"] for r in mode_results)
        efficiency = successes / max(total_llm, 1)
        print(f"  {mode}: {efficiency:.3f} successes per LLM call")
    
    # Winner determination
    print("\n" + "=" * 100)
    print("WINNER BY DIFFICULTY")
    print("=" * 100)
    
    for diff in ["easy", "medium", "hard", "very_hard"]:
        print(f"\n{diff.upper()}:")
        best_mode = None
        best_success = -1
        best_efficiency = -1
        
        for mode in modes:
            mode_results = [r for r in all_results if r["mode"] == mode and r["difficulty"] == diff]
            if mode_results:
                successes = sum(1 for r in mode_results if r["success"])
                total_llm = sum(r["llm_calls"] for r in mode_results)
                efficiency = successes / max(total_llm, 1)
                
                if successes > best_success or (successes == best_success and efficiency > best_efficiency):
                    best_success = successes
                    best_efficiency = efficiency
                    best_mode = mode
        
        if best_mode:
            print(f"  🏆 Winner: {best_mode.upper()} ({best_success} successes)")
    
    # Save results
    output_path = Path(__file__).parent.parent / "research" / f"comprehensive_test_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
