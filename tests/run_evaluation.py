#!/usr/bin/env python
"""
QAgents-Workflows: Main Evaluation Runner
Runs comparative tests between Blackboard, Guided, and Naked modes.

Usage:
    python run_evaluation.py                    # Run all tests
    python run_evaluation.py --mode naked       # Test specific mode
    python run_evaluation.py --problem easy_001 # Test specific problem
    python run_evaluation.py --quick            # Quick test (1 run per problem)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config, set_mode
from client import get_client
from tests import (
    EvaluationHarness, 
    ALL_PROBLEMS, 
    EASY_PROBLEMS,
    get_problem
)


def setup_logging(verbose: bool = True):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def check_mcp_server():
    """Check if MCP server is running."""
    client = get_client()
    if not client.health_check():
        print("\n❌ ERROR: QuantumArchitect-MCP server is not running!")
        print("\nPlease start it with:")
        print("  cd D:\\teach\\quantum-circuits")
        print("  & .venv\\Scripts\\Activate.ps1")
        print("  python QuantumArchitect-MCP\\app.py")
        print()
        return False
    print("✅ MCP server is running")
    return True


def run_quick_test():
    """Run a quick sanity test."""
    print("\n Running Quick Test (Naked mode, Bell State)")
    print("-" * 50)
    
    from orchestrators import create_orchestrator
    from tests import BELL_STATE_PROBLEM
    
    orchestrator = create_orchestrator("naked")
    result = orchestrator.run(BELL_STATE_PROBLEM.goal)
    
    print(f"Success: {result.success}")
    print(f"Time: {result.execution_time_ms:.1f}ms")
    print(f"Steps: {result.steps_completed}")
    
    if result.final_output:
        print(f"\nGenerated Circuit:")
        print(result.final_output[:500] if len(result.final_output) > 500 else result.final_output)
        
    if result.errors:
        print(f"\nErrors: {result.errors}")
        
    return result.success


def run_full_evaluation(problems=None, modes=None, num_runs=3):
    """Run full comparative evaluation."""
    print("\n Starting Full Evaluation")
    print("=" * 60)

    if problems is None:
        problems = EASY_PROBLEMS  # Start with easy problems
    if modes is None:
        modes = ["blackboard", "guided", "naked"]

    print(f"Problems: {len(problems)}")
    print(f"Modes: {modes}")
    print(f"Runs per problem: {num_runs}")
    print()

    harness = EvaluationHarness(num_runs=num_runs)

    try:
        results = harness.evaluate_all(problems=problems, modes=modes)

        # Generate and print report
        report = harness.generate_report()
        print("\n" + report)

        # Save report to file
        report_path = Path(__file__).parent / "evaluation_report.txt"
        report_path.write_text(report)
        print(f"\n Report saved to: {report_path}")

        # Export CSV for research
        csv_path = harness.export_csv()
        print(f" CSV exported to: {csv_path}")

        # Print summary stats
        stats = harness.get_summary_stats()
        print("\n Summary Statistics:")
        for mode, mode_stats in stats.get('modes', {}).items():
            print(f"  {mode}: {mode_stats['success_rate']*100:.1f}% success, "
                  f"{mode_stats['total_llm_requests']} LLM calls, "
                  f"{mode_stats['total_tokens']} tokens")

        return True

    except Exception as e:
        logging.exception(f"Evaluation failed: {e}")
        return False
def main():
    parser = argparse.ArgumentParser(
        description="QAgents Comparative Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                    # Full evaluation
  python run_evaluation.py --quick            # Quick sanity test
  python run_evaluation.py --mode naked       # Test naked mode only
  python run_evaluation.py --easy             # Only easy problems
  python run_evaluation.py --runs 10          # 10 runs per problem
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="Run quick sanity test only")
    parser.add_argument("--mode", choices=["blackboard", "guided", "naked"],
                        help="Test specific mode only")
    parser.add_argument("--problem", type=str,
                        help="Test specific problem by ID")
    parser.add_argument("--easy", action="store_true",
                        help="Only easy problems")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per problem (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("=" * 60)
    print("[EVALUATION] QAgents-Workflows Comparative Evaluation")
    print("=" * 60)
    
    # Check MCP server
    if not check_mcp_server():
        sys.exit(1)
        
    # Quick test mode
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
        
    # Determine problems to run
    if args.problem:
        problem = get_problem(args.problem)
        if not problem:
            print(f"❌ Unknown problem: {args.problem}")
            sys.exit(1)
        problems = [problem]
    elif args.easy:
        problems = EASY_PROBLEMS
    else:
        problems = ALL_PROBLEMS
        
    # Determine modes to test
    modes = [args.mode] if args.mode else None
    
    # Run evaluation
    success = run_full_evaluation(
        problems=problems,
        modes=modes,
        num_runs=args.runs
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
