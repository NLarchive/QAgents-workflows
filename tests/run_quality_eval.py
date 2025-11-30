# Path: QAgents-workflos/run_quality_eval.py
# Relations: Uses tests/quality_evaluation_harness.py, database/circuit_quality_db.py
# Description: CLI entry point for quality-focused evaluation
#              Run with: python run_quality_eval.py --mode all --difficulty easy
#              Generates quality comparison report with actual QASM circuits

"""
Quality Evaluation Runner: CLI entry point for circuit quality comparison.

Usage:
    python run_quality_eval.py --mode all --difficulty easy
    python run_quality_eval.py --mode naked --problem easy_001
    python run_quality_eval.py --report RUN_ID
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure API key is set BEFORE importing config
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

from tests.quality_evaluation_harness import QualityEvaluationHarness, run_quick_quality_test
from tests.test_problems import get_problem, get_problems_by_difficulty
from database.circuit_quality_db import get_quality_db
from config import set_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly set API key in config after logging is ready
if api_key:
    set_api_key(api_key)
    logger.info(f"API Key configured: {api_key[:10]}...")
else:
    logger.warning("No GOOGLE_API_KEY or GENAI_API_KEY found in environment")


def run_evaluation(args):
    """Run quality evaluation based on arguments."""
    harness = QualityEvaluationHarness()

    # Parse modes
    if args.mode == 'all':
        modes = ['naked', 'guided', 'blackboard']
    else:
        modes = [args.mode]

    # Parse difficulties
    if args.difficulty == 'all':
        difficulties = ['easy', 'medium', 'hard']
    else:
        difficulties = [args.difficulty]

    # Check if specific problem
    if args.problem:
        problem = get_problem(args.problem)
        if not problem:
            print(f"ERROR: Problem not found: {args.problem}")
            return

        print(f"\n{'='*60}")
        print(f"Running quality evaluation for: {args.problem}")
        print(f"Modes: {modes}")
        print(f"{'='*60}\n")

        results = harness.evaluate_problem_all_modes(problem, modes)

        # Print results
        for mode, result in results.items():
            print(f"\n{mode.upper()}:")
            print(f"  Success: {'✅' if result.success else '❌'}")
            print(f"  Quality Score: {result.quality_metrics.overall_score()}/100")
            print(f"  Depth: {result.quality_metrics.depth}")
            print(f"  Gates: {result.quality_metrics.gate_count}")
            print(f"  CX: {result.quality_metrics.cx_count}")
            print(f"  Time: {result.execution_time_ms:.0f}ms")
            print(f"  LLM Calls: {result.llm_requests}")
            if result.qasm_code:
                print(f"  QASM ({len(result.qasm_code)} chars):")
                lines = result.qasm_code.split('\n')[:10]
                for line in lines:
                    print(f"    {line}")
                if len(result.qasm_code.split('\n')) > 10:
                    print("    ...")
    else:
        # Full evaluation
        print(f"\n{'='*60}")
        print(f"Running full quality evaluation")
        print(f"Difficulties: {difficulties}")
        print(f"Modes: {modes}")
        print(f"Max problems: {args.max_problems or 'all'}")
        print(f"{'='*60}\n")

        run_id = harness.run_full_evaluation(
            difficulties=difficulties,
            modes=modes,
            max_problems=args.max_problems
        )

        # Print summary
        harness.print_summary(run_id)

        # Generate report file
        report = harness.generate_report(run_id)
        report_path = Path(__file__).parent / f"QUALITY_REPORT_{run_id}.md"
        report_path.write_text(report, encoding='utf-8')
        print(f"\nFull report saved to: {report_path}")

        print(f"\nRun ID: {run_id}")
        print("Use --report <run_id> to regenerate report later")


def show_report(run_id: str):
    """Show report for a specific run."""
    harness = QualityEvaluationHarness()
    harness.run_id = run_id  # Set to existing run

    report = harness.generate_report(run_id)
    print(report)


def list_runs():
    """List all evaluation runs."""
    db = get_quality_db()

    query = "SELECT run_id, timestamp, description, num_problems FROM comparison_runs ORDER BY timestamp DESC LIMIT 20"
    import sqlite3
    with sqlite3.connect(db.db_file) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()

    if not rows:
        print("No evaluation runs found.")
        return

    print("\nRecent Evaluation Runs:")
    print("-" * 80)
    for row in rows:
        print(f"{row['run_id']} | {row['timestamp']} | {row['num_problems']} problems | {row['description'] or 'N/A'}")
    print("-" * 80)


def quick_test(args):
    """Run a quick single test."""
    mode = args.mode if args.mode != 'all' else 'naked'
    problem_id = args.problem or 'easy_001'

    print(f"\nQuick test: {problem_id} with {mode} mode")
    print("-" * 40)

    try:
        result = run_quick_quality_test(mode, problem_id)
        print(f"Success: {'✅' if result.success else '❌'}")
        print(f"Quality Score: {result.quality_metrics.overall_score()}/100")
        print(f"Depth: {result.quality_metrics.depth}")
        print(f"Gates: {result.quality_metrics.gate_count}")
        if result.qasm_code:
            print(f"\nQASM:\n{result.qasm_code[:500]}")
        if result.errors:
            print(f"\nErrors: {result.errors}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Quality-focused quantum circuit evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_quality_eval.py --quick                     # Quick test
  python run_quality_eval.py --mode all --difficulty easy
  python run_quality_eval.py --problem easy_001 --mode all
  python run_quality_eval.py --list                      # List previous runs
  python run_quality_eval.py --report quality_20241128_120000
"""
    )

    parser.add_argument('--mode', choices=['naked', 'guided', 'blackboard', 'all'],
                        default='all', help='Orchestration mode(s) to test')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard', 'all'],
                        default='easy', help='Problem difficulty level(s)')
    parser.add_argument('--problem', type=str, help='Specific problem ID to test')
    parser.add_argument('--max-problems', type=int, help='Maximum problems to test')
    parser.add_argument('--quick', action='store_true', help='Run quick single test')
    parser.add_argument('--report', type=str, help='Generate report for run ID')
    parser.add_argument('--list', action='store_true', help='List previous runs')

    args = parser.parse_args()

    if args.list:
        list_runs()
    elif args.report:
        show_report(args.report)
    elif args.quick:
        quick_test(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
