# Path: QAgents-workflos/tests/quality_evaluation_harness.py
# Relations: Uses orchestrators/, tests/circuit_quality_analyzer.py, database/circuit_quality_db.py
# Description: Quality-focused evaluation harness that stores QASM circuits
#              Runs all 3 modes, measures quality via MCP, stores in database
#              Generates comparison reports with actual circuit outputs

"""
Quality Evaluation Harness: Run evaluations focused on CIRCUIT QUALITY.
Key difference from regular harness: stores actual QASM and measures quality.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

from .test_problems import TestProblem, ALL_PROBLEMS, get_problem, get_problems_by_difficulty, ProblemDifficulty
from .circuit_quality_analyzer import CircuitQualityAnalyzer, AnalysisResult
from database.circuit_quality_db import (
    CircuitQualityDB, CircuitEvaluation, QualityMetrics, get_quality_db
)

logger = logging.getLogger(__name__)


class QualityEvaluationHarness:
    """
    Runs quality-focused evaluations across all orchestration modes.
    PRIMARY FOCUS: Circuit quality, not just success rate.
    STORES: Full QASM code in database for later analysis.
    """
    
    def __init__(self, mcp_url: str = "http://127.0.0.1:7861"):
        self.mcp_url = mcp_url
        self.analyzer = CircuitQualityAnalyzer(mcp_url)
        self.db = get_quality_db()
        self.run_id = f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def evaluate_single(self, problem: TestProblem, mode: str) -> CircuitEvaluation:
        """
        Run a single evaluation and return full CircuitEvaluation with QASM.
        
        Args:
            problem: The test problem to solve
            mode: 'naked', 'guided', or 'blackboard'
        
        Returns:
            CircuitEvaluation with full QASM and quality metrics
        """
        from orchestrators import create_orchestrator
        
        logger.info(f"Evaluating {problem.id} with {mode} mode")
        
        # Reset cost tracking
        try:
            from config import reset_cost_tracking, get_cost_summary
            reset_cost_tracking()
        except ImportError:
            get_cost_summary = lambda: {}
        
        # Initialize result
        eval_result = CircuitEvaluation(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            problem_id=problem.id,
            problem_goal=problem.goal,
            mode=mode
        )
        
        start_time = time.perf_counter()
        
        try:
            # Create and run orchestrator
            orchestrator = create_orchestrator(mode)
            result = orchestrator.run(problem.goal)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            eval_result.execution_time_ms = elapsed_ms
            
            # Extract QASM
            qasm = result.final_output
            if isinstance(qasm, list):
                qasm = qasm[0] if qasm else None
            if qasm is not None:
                qasm = str(qasm) if not isinstance(qasm, str) else qasm
            
            eval_result.qasm_code = qasm or ""
            eval_result.success = result.success and bool(qasm)
            
            if not eval_result.success:
                eval_result.errors = result.errors
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            eval_result.execution_time_ms = elapsed_ms
            eval_result.success = False
            eval_result.errors = [str(e)]
            logger.error(f"Evaluation failed for {problem.id}/{mode}: {e}")
        
        # Get cost metrics
        try:
            cost = get_cost_summary()
            eval_result.llm_requests = cost.get('total_requests', 0)
            eval_result.tokens_used = cost.get('total_tokens', 0)
        except Exception:
            pass
        
        # Analyze quality if we have QASM
        if eval_result.qasm_code:
            expected = problem.expected.expected_states if problem.expected else None
            analysis = self.analyzer.analyze_circuit(eval_result.qasm_code, expected)
            
            eval_result.quality_metrics = QualityMetrics(
                depth=analysis.depth,
                gate_count=analysis.gate_count,
                cx_count=analysis.cx_count,
                single_qubit_count=analysis.single_qubit_count,
                hardware_fitness=analysis.hardware_fitness,
                syntax_valid=analysis.syntax_valid,
                state_correctness=analysis.state_correctness,
                complexity_score=analysis.complexity_score,
                noise_estimate=analysis.noise_estimate
            )
            
            if analysis.errors:
                eval_result.errors.extend(analysis.errors)
        
        # Store in database
        eval_id = self.db.save_evaluation(eval_result)
        eval_result.id = eval_id
        
        logger.info(f"Stored evaluation {eval_id}: {problem.id}/{mode} - "
                   f"success={eval_result.success}, score={eval_result.quality_metrics.overall_score()}")
        
        return eval_result
    
    def evaluate_problem_all_modes(self, problem: TestProblem, 
                                   modes: List[str] = None) -> Dict[str, CircuitEvaluation]:
        """Evaluate a single problem with all modes."""
        if modes is None:
            modes = ['naked', 'guided', 'blackboard']
        
        results = {}
        for mode in modes:
            results[mode] = self.evaluate_single(problem, mode)
        
        return results
    
    def run_full_evaluation(self, 
                           difficulties: List[str] = None,
                           modes: List[str] = None,
                           max_problems: int = None) -> str:
        """
        Run full evaluation across problems and modes.
        
        Args:
            difficulties: List of difficulties to test ('easy', 'medium', 'hard')
            modes: List of modes to test ('naked', 'guided', 'blackboard')
            max_problems: Maximum number of problems to test (for quick runs)
        
        Returns:
            run_id for this evaluation run
        """
        if difficulties is None:
            difficulties = ['easy', 'medium', 'hard']
        if modes is None:
            modes = ['naked', 'guided', 'blackboard']
        
        # Gather problems
        all_probs = []
        for diff in difficulties:
            # Convert string to enum if needed
            if isinstance(diff, str):
                try:
                    diff_enum = ProblemDifficulty(diff)
                except ValueError:
                    logger.warning(f"Invalid difficulty: {diff}")
                    continue
            else:
                diff_enum = diff
                
            probs = get_problems_by_difficulty(diff_enum)
            all_probs.extend(probs)
        
        if max_problems:
            all_probs = all_probs[:max_problems]
        
        logger.info(f"Starting quality evaluation run {self.run_id}")
        logger.info(f"Problems: {len(all_probs)}, Modes: {modes}")
        
        # Run evaluations
        total = len(all_probs) * len(modes)
        completed = 0
        
        for problem in all_probs:
            for mode in modes:
                try:
                    self.evaluate_single(problem, mode)
                    completed += 1
                    logger.info(f"Progress: {completed}/{total}")
                except Exception as e:
                    logger.error(f"Failed {problem.id}/{mode}: {e}")
                    completed += 1
        
        # Save run summary
        summary = self.db.get_quality_summary(self.run_id)
        self.db.save_comparison_run(
            run_id=self.run_id,
            description=f"Quality evaluation: {len(all_probs)} problems, {modes}",
            num_problems=len(all_probs),
            modes=modes,
            summary=summary
        )
        
        return self.run_id
    
    def generate_report(self, run_id: Optional[str] = None) -> str:
        """Generate a comprehensive quality comparison report."""
        if run_id is None:
            run_id = self.run_id
        
        # Get summary
        summary = self.db.get_quality_summary(run_id)
        
        # Get full circuit export
        circuits_md = self.db.export_circuits_markdown(run_id)
        
        # Build report
        report = []
        report.append("# CIRCUIT QUALITY EVALUATION REPORT\n")
        report.append(f"Run ID: {run_id}\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")
        
        report.append("## EXECUTIVE SUMMARY\n\n")
        
        # Summary table
        report.append("| Mode | Success Rate | Quality Score | Avg Depth | Avg Gates | Avg CX | HW Fitness | LLM Calls |\n")
        report.append("|------|-------------|---------------|-----------|-----------|--------|------------|----------|\n")
        
        for mode in ['naked', 'guided', 'blackboard']:
            if mode in summary.get('modes', {}):
                m = summary['modes'][mode]
                report.append(
                    f"| {mode.upper()} | {m['success_rate']*100:.0f}% | "
                    f"{m['avg_quality_score']:.1f}/100 | {m['avg_depth']:.1f} | "
                    f"{m['avg_gates']:.1f} | {m['avg_cx_count']:.1f} | "
                    f"{m['avg_hardware_fitness']:.3f} | {m['total_llm_requests']} |\n"
                )
        
        report.append("\n## KEY FINDINGS\n\n")
        
        # Determine winner
        modes_data = summary.get('modes', {})
        if modes_data:
            best_quality = max(modes_data.items(), key=lambda x: x[1].get('avg_quality_score', 0))
            best_success = max(modes_data.items(), key=lambda x: x[1].get('success_rate', 0))
            lowest_cost = min(modes_data.items(), key=lambda x: x[1].get('total_llm_requests', float('inf')))
            
            report.append(f"- **Best Quality**: {best_quality[0].upper()} ({best_quality[1]['avg_quality_score']:.1f}/100)\n")
            report.append(f"- **Best Success Rate**: {best_success[0].upper()} ({best_success[1]['success_rate']*100:.0f}%)\n")
            report.append(f"- **Lowest Cost**: {lowest_cost[0].upper()} ({lowest_cost[1]['total_llm_requests']} LLM calls)\n")
            
            # Quality per LLM call
            report.append("\n### Quality Efficiency (Quality Score per LLM Call)\n\n")
            for mode, data in modes_data.items():
                llm_calls = data.get('total_llm_requests', 1) or 1
                quality = data.get('avg_quality_score', 0)
                efficiency = quality / llm_calls
                report.append(f"- {mode.upper()}: {efficiency:.2f} quality points per LLM call\n")
        
        report.append("\n---\n")
        report.append("\n## DETAILED CIRCUIT COMPARISONS\n")
        report.append(circuits_md)
        
        return "".join(report)
    
    def print_summary(self, run_id: Optional[str] = None):
        """Print a quick summary to console."""
        if run_id is None:
            run_id = self.run_id
        
        summary = self.db.get_quality_summary(run_id)
        
        print("\n" + "="*70)
        print("QUALITY EVALUATION SUMMARY")
        print("="*70)
        
        modes = summary.get('modes', {})
        for mode in ['naked', 'guided', 'blackboard']:
            if mode in modes:
                m = modes[mode]
                print(f"\n{mode.upper()}:")
                print(f"  Success Rate:    {m['success_rate']*100:.0f}%")
                print(f"  Quality Score:   {m['avg_quality_score']:.1f}/100")
                print(f"  Avg Depth:       {m['avg_depth']:.1f}")
                print(f"  Avg Gates:       {m['avg_gates']:.1f}")
                print(f"  Avg CX Count:    {m['avg_cx_count']:.1f}")
                print(f"  HW Fitness:      {m['avg_hardware_fitness']:.3f}")
                print(f"  LLM Requests:    {m['total_llm_requests']}")
        
        print("\n" + "="*70)


def run_quick_quality_test(mode: str = 'naked', problem_id: str = 'bell_state') -> CircuitEvaluation:
    """Quick test function to verify system works."""
    problem = get_problem(problem_id)
    if not problem:
        raise ValueError(f"Problem not found: {problem_id}")
    
    harness = QualityEvaluationHarness()
    return harness.evaluate_single(problem, mode)
