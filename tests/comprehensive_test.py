# Path: QAgents-workflos/tests/comprehensive_test.py
# Relations: Uses orchestrators/, tests/test_problems.py, config.py
# Description: Comprehensive test across all difficulties with detailed diagnostics
#              Run with: python tests/comprehensive_test.py

"""
Comprehensive Circuit Generation Test

Tests all 9 problems (easy, medium, hard) with all 3 modes (naked, guided, blackboard).
Provides detailed diagnostics on where each mode succeeds/fails.
"""

import sys
import time
import os
from datetime import datetime
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_problems import ALL_PROBLEMS, ProblemDifficulty
from orchestrators import create_orchestrator
from config import reset_cost_tracking, get_cost_summary, set_api_key


def extract_qasm(result):
    """Extract QASM from orchestrator result."""
    if not result or not result.final_output:
        return None
    
    qasm = result.final_output
    if isinstance(qasm, list):
        qasm = qasm[0] if qasm else None
    
    return str(qasm) if qasm else None


def validate_qasm(qasm):
    """Validate QASM structure and count gates."""
    if not qasm:
        return {"valid": False, "has_qreg": False, "gate_count": 0, "depth": 0}
    
    valid = "OPENQASM" in qasm
    has_qreg = "qreg" in qasm
    
    # Count gates
    gate_count = 0
    for gate in ['h ', 'h(', 'x ', 'x(', 'z ', 'z(', 'cx ', 'cx(', 'cz ', 
                 'swap ', 't ', 's ', 'ry(', 'rz(', 'rx(', 'u1(', 'u2(', 'u3(']:
        gate_count += qasm.lower().count(gate)
    
    # Estimate depth (simplified)
    lines = [l for l in qasm.split('\n') if l.strip() and not l.strip().startswith('//')]
    depth = len([l for l in lines if any(g in l.lower() for g in ['h ', 'x ', 'cx ', 'cz ', 'swap'])])
    
    return {"valid": valid, "has_qreg": has_qreg, "gate_count": gate_count, "depth": depth}


def run_comprehensive_test():
    """Run comprehensive test across all problems and modes."""
    
    # Set API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GENAI_API_KEY')
    if api_key:
        set_api_key(api_key)
    else:
        print("ERROR: No API key found. Set GOOGLE_API_KEY environment variable.")
        return
    
    print("=" * 100)
    print("COMPREHENSIVE CIRCUIT GENERATION TEST - ALL DIFFICULTIES")
    print("=" * 100)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Problems: {len(ALL_PROBLEMS)} total (3 easy, 3 medium, 3 hard)")
    print(f"Modes: naked, guided, blackboard")
    print("=" * 100)
    
    # Store all results
    all_results = []
    
    # Test each problem with each mode
    for problem in ALL_PROBLEMS:
        print(f"\n\n{'=' * 100}")
        print(f"PROBLEM: {problem.id} - {problem.name}")
        print(f"Difficulty: {problem.difficulty.value.upper()}")
        print(f"Category: {problem.category.value}")
        print(f"Expected qubits: {problem.expected.min_qubits}-{problem.expected.max_qubits}")
        print(f"Required gates: {problem.expected.required_gates}")
        print(f"Expected states: {problem.expected.expected_states}")
        print("=" * 100)
        
        for mode in ['naked', 'guided', 'blackboard']:
            print(f"\n--- {mode.upper()} MODE ---")
            reset_cost_tracking()
            
            start = time.perf_counter()
            result = None
            qasm = None
            
            try:
                orchestrator = create_orchestrator(mode)
                result = orchestrator.run(problem.goal)
                
                elapsed = (time.perf_counter() - start) * 1000
                cost = get_cost_summary()
                
                # Extract and validate QASM
                qasm = extract_qasm(result)
                validation = validate_qasm(qasm)
                
                success = result.success if result else False
                errors = result.errors if result else []
                
                # Print detailed results
                status = '✅' if success and validation['valid'] else '❌'
                print(f"{status} Success: {success}")
                print(f"   Time: {elapsed:.0f}ms")
                print(f"   LLM Calls: {cost.get('total_requests', 0)}")
                print(f"   Tokens: {cost.get('total_tokens', 0)}")
                print(f"   QASM Valid: {validation['valid']}")
                print(f"   Has qreg: {validation['has_qreg']}")
                print(f"   Gate Count: {validation['gate_count']}")
                print(f"   Est. Depth: {validation['depth']}")
                
                if errors:
                    print(f"   ⚠️  Errors: {errors[:2]}")
                
                if qasm:
                    # Show first few lines of QASM
                    lines = qasm.split('\n')[:8]
                    print("   QASM:")
                    for line in lines:
                        print(f"      {line}")
                    if len(qasm.split('\n')) > 8:
                        print("      ...")
                else:
                    print("   QASM: None generated")
                
                all_results.append({
                    'problem_id': problem.id,
                    'problem_name': problem.name,
                    'difficulty': problem.difficulty.value,
                    'category': problem.category.value,
                    'mode': mode,
                    'success': success and validation['valid'],
                    'qasm_valid': validation['valid'],
                    'time_ms': elapsed,
                    'llm_calls': cost.get('total_requests', 0),
                    'tokens': cost.get('total_tokens', 0),
                    'gate_count': validation['gate_count'],
                    'depth': validation['depth'],
                    'qasm': qasm[:500] if qasm else None,
                    'error': str(errors[0])[:100] if errors else None
                })
                
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"❌ EXCEPTION: {error_msg}")
                
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    'problem_id': problem.id,
                    'problem_name': problem.name,
                    'difficulty': problem.difficulty.value,
                    'category': problem.category.value,
                    'mode': mode,
                    'success': False,
                    'qasm_valid': False,
                    'time_ms': elapsed,
                    'llm_calls': 0,
                    'tokens': 0,
                    'gate_count': 0,
                    'depth': 0,
                    'qasm': None,
                    'error': error_msg[:100]
                })
    
    # Print final summary
    print_summary(all_results)
    
    # Save results to JSON
    output_path = Path(__file__).parent.parent / f"research/comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")
    
    return all_results


def print_summary(all_results):
    """Print summary by difficulty and mode."""
    
    print("\n\n" + "=" * 100)
    print("FINAL SUMMARY BY DIFFICULTY AND MODE")
    print("=" * 100)
    
    for diff in ['easy', 'medium', 'hard']:
        print(f"\n{diff.upper()} PROBLEMS:")
        print("-" * 80)
        
        for mode in ['naked', 'guided', 'blackboard']:
            mode_results = [r for r in all_results if r['difficulty'] == diff and r['mode'] == mode]
            if mode_results:
                successes = sum(1 for r in mode_results if r['success'])
                total = len(mode_results)
                avg_time = sum(r['time_ms'] for r in mode_results) / total
                total_llm = sum(r['llm_calls'] for r in mode_results)
                avg_gates = sum(r['gate_count'] for r in mode_results) / total
                
                status = '✅' if successes == total else '⚠️ ' if successes > 0 else '❌'
                print(f"{status} {mode:12} | Success: {successes}/{total} | Time: {avg_time:>6.0f}ms | LLM: {total_llm:>2} | Avg Gates: {avg_gates:.1f}")
                
                # Show failures
                failures = [r for r in mode_results if not r['success']]
                for f in failures:
                    error_msg = f['error'][:60] if f['error'] else 'No QASM generated'
                    print(f"      ❌ {f['problem_id']}: {error_msg}")
    
    # Calculate winners
    print("\n\n" + "=" * 100)
    print("🏆 WINNER BY DIFFICULTY (Score = Success*100 - Time/1000 - LLM*0.5)")
    print("=" * 100)
    
    for diff in ['easy', 'medium', 'hard']:
        print(f"\n{diff.upper()}:")
        best_mode = None
        best_score = -999
        
        for mode in ['naked', 'guided', 'blackboard']:
            mode_results = [r for r in all_results if r['difficulty'] == diff and r['mode'] == mode]
            if mode_results:
                successes = sum(1 for r in mode_results if r['success'])
                total = len(mode_results)
                avg_time = sum(r['time_ms'] for r in mode_results) / total
                total_llm = sum(r['llm_calls'] for r in mode_results)
                
                success_rate = successes / total
                time_penalty = avg_time / 1000
                llm_penalty = total_llm * 0.5
                score = success_rate * 100 - time_penalty - llm_penalty
                
                print(f"  {mode:12}: Score={score:>6.1f} (Success={success_rate*100:.0f}%, Time={avg_time:.0f}ms, LLM={total_llm})")
                
                if score > best_score:
                    best_score = score
                    best_mode = mode
        
        print(f"  🏆 WINNER: {best_mode.upper() if best_mode else 'NONE'}")
    
    # Overall recommendation
    print("\n\n" + "=" * 100)
    print("OVERALL RECOMMENDATIONS")
    print("=" * 100)
    
    # Calculate overall stats per mode
    for mode in ['naked', 'guided', 'blackboard']:
        mode_results = [r for r in all_results if r['mode'] == mode]
        if mode_results:
            successes = sum(1 for r in mode_results if r['success'])
            total = len(mode_results)
            avg_time = sum(r['time_ms'] for r in mode_results) / total
            total_llm = sum(r['llm_calls'] for r in mode_results)
            avg_gates = sum(r['gate_count'] for r in mode_results) / total
            
            print(f"\n{mode.upper()}:")
            print(f"  Overall Success: {successes}/{total} ({100*successes/total:.0f}%)")
            print(f"  Average Time: {avg_time:.0f}ms")
            print(f"  Total LLM Calls: {total_llm}")
            print(f"  Average Gates: {avg_gates:.1f}")
            
            # List failures
            failures = [r for r in mode_results if not r['success']]
            if failures:
                print(f"  Failures ({len(failures)}):")
                for f in failures:
                    print(f"    - {f['problem_id']} ({f['difficulty']}): {f['error'][:50] if f['error'] else 'Unknown'}")


if __name__ == "__main__":
    run_comprehensive_test()
