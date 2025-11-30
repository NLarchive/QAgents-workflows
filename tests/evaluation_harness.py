# Path: QAgents-workflos/tests/evaluation_harness.py
# Relations: Uses orchestrators, tools, database, config modules
#            Uses agents/llm_adapter.py for LLM usage tracking
# Description: Evaluation harness for comparative testing of Blackboard, Guided, and Naked modes
#              Includes cost tracking (requests, tokens, time) for each mode
#              Exports results to CSV for research analysis
"""
Evaluation Harness: Measure time, quality, effectiveness, reliability.
Runs comparative tests across Blackboard, Guided, and Naked modes.

COST TRACKING METRICS:
======================
For each mode, tracks:
  - LLM requests: Number of calls to LLM API
  - Tokens used: Total tokens consumed (input + output)
  - Time: Total execution time
  - Quality: Circuit correctness and complexity scores
  
MODES:
======
  - Naked: Direct LLM (1 call/problem) - baseline test
  - Guided: Structured workflow (4 LLM calls/problem)
  - Blackboard: Free-form collaboration (8-12 LLM calls/problem)

OUTPUT FORMATS:
===============
  - TXT: Human-readable report
  - CSV: Research data for longitudinal analysis
"""

import time
import json
import csv
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from .test_problems import TestProblem, ALL_PROBLEMS, get_problem
from database import get_database, ResultEntry

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result for a single metric."""
    name: str
    value: float
    unit: str
    passed: bool = True
    details: str = ""


@dataclass
class CostMetrics:
    """Cost metrics for a single run."""
    llm_requests: int = 0
    mcp_requests: int = 0
    tokens_used: int = 0
    time_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    
    def cost_per_quality(self, quality_score: float) -> float:
        """Calculate cost-per-quality ratio (lower is better)."""
        if quality_score <= 0:
            return float('inf')
        # Cost = (requests * 1) + (tokens / 1000) + (time_ms / 1000)
        cost = self.llm_requests + (self.tokens_used / 1000) + (self.time_ms / 1000)
        return cost / quality_score


@dataclass
class EvaluationResult:
    """Result of evaluating a single run."""
    problem_id: str
    system_mode: str
    run_number: int
    success: bool
    execution_time_ms: float
    circuit_qasm: Optional[str]
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    cost_metrics: CostMetrics = field(default_factory=CostMetrics)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedResults:
    """Aggregated results for a problem across all runs."""
    problem_id: str
    system_mode: str
    num_runs: int
    success_rate: float
    avg_time_ms: float
    std_time_ms: float
    avg_quality_score: float
    effectiveness: float
    reliability: float
    # Cost aggregates
    total_llm_requests: int = 0
    total_mcp_requests: int = 0
    total_tokens: int = 0
    avg_cost_per_quality: float = 0.0
    all_results: List[EvaluationResult] = field(default_factory=list)


class EvaluationHarness:
    """
    Runs comparative evaluations across different orchestration modes.
    Measures: Time, Quality, Effectiveness, Reliability, Cost
    """

    def __init__(self, num_runs: int = 5, timeout_seconds: float = 120.0):
        self.num_runs = num_runs
        self.timeout_seconds = timeout_seconds
        self.db = get_database()
        self.results: Dict[str, Dict[str, AggregatedResults]] = {}
        
        # Track MCP requests per run
        self._mcp_request_count = 0

    def _reset_cost_tracking(self):
        """Reset cost tracking before a run."""
        try:
            from config import reset_cost_tracking
            reset_cost_tracking()
        except Exception:
            pass
        self._mcp_request_count = 0
    
    def _get_cost_summary(self) -> Dict:
        """Get cost tracking summary after a run."""
        try:
            from config import get_cost_summary
            return get_cost_summary()
        except Exception:
            return {"total_requests": 0, "total_tokens": 0, "total_time_ms": 0.0}
    
    def _get_llm_usage_summary(self) -> Dict:
        """Get LLM usage from rate limiter."""
        try:
            from agents.llm_adapter import get_usage_summary
            return get_usage_summary()
        except Exception:
            return {}

    def evaluate_single_run(self, problem: TestProblem, mode: str,
                            run_number: int) -> EvaluationResult:
        """Run a single evaluation with cost tracking."""
        from orchestrators import create_orchestrator
        from tools import invoke_tool

        logger.info(f"Running {mode} on {problem.id}, run {run_number}")

        # Reset cost tracking
        self._reset_cost_tracking()
        
        errors = []
        circuit_qasm = None
        metrics = {}
        success = False
        cost_metrics = CostMetrics()

        start_time = time.perf_counter()

        try:
            # Create and run orchestrator
            orchestrator = create_orchestrator(mode)
            result = orchestrator.run(problem.goal)

            circuit_qasm = result.final_output

            # Handle list responses from MCP
            if isinstance(circuit_qasm, list):
                circuit_qasm = circuit_qasm[0] if circuit_qasm else None

            # Ensure it's a string or None
            if circuit_qasm is not None:
                circuit_qasm = str(circuit_qasm) if not isinstance(circuit_qasm, str) else circuit_qasm

            success = result.success and circuit_qasm is not None

            if not success:
                errors.extend(result.errors)

        except Exception as e:
            success = False
            errors.append(str(e))
            logger.error(f"Evaluation failed: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Collect cost metrics
        cost_summary = self._get_cost_summary()
        llm_usage = self._get_llm_usage_summary()
        
        cost_metrics = CostMetrics(
            llm_requests=cost_summary.get("total_requests", 0),
            mcp_requests=self._mcp_request_count,
            tokens_used=cost_summary.get("total_tokens", 0),
            time_ms=elapsed_ms,
            models_used=list(cost_summary.get("model_breakdown", {}).keys())
        )

        # Calculate metrics if we have a circuit
        if circuit_qasm:
            metrics = self._calculate_metrics(circuit_qasm, problem)

        return EvaluationResult(
            problem_id=problem.id,
            system_mode=mode,
            run_number=run_number,
            success=success,
            execution_time_ms=elapsed_ms,
            circuit_qasm=circuit_qasm,
            metrics=metrics,
            cost_metrics=cost_metrics,
            errors=errors
        )

    def _calculate_metrics(self, qasm: str, problem: TestProblem) -> Dict[str, MetricResult]:
        """Calculate quality metrics for a circuit."""
        from tools import invoke_tool

        metrics = {}

        try:
            # Helper to extract value from potentially nested result
            def extract_value(result, key, default=0):
                val = result.get(key, default)
                if isinstance(val, dict):
                    return val.get('depth', val.get('value', val.get('score', default)))
                elif isinstance(val, list):
                    return val[0] if val else default
                return val

            # 1. Depth metric
            self._mcp_request_count += 1
            depth_result = invoke_tool("get_circuit_depth", qasm=qasm)
            if depth_result.get("success"):
                depth = extract_value(depth_result, "depth", 0)
                if isinstance(depth, dict):
                    depth = depth.get('depth', 0)
                max_depth = problem.expected.max_depth or 100
                passed = depth <= max_depth if max_depth else True
                metrics["depth"] = MetricResult(
                    name="Circuit Depth",
                    value=float(depth) if depth else 0,
                    unit="layers",
                    passed=passed,
                    details=f"Expected max: {max_depth}"
                )

            # 2. Complexity score
            self._mcp_request_count += 1
            complexity_result = invoke_tool("calculate_complexity", qasm=qasm)
            if complexity_result.get("success"):
                score = complexity_result.get("score", {})
                if isinstance(score, dict):
                    complexity_value = score.get("complexity_score", score.get("total", 0))
                elif isinstance(score, list):
                    complexity_value = 0
                else:
                    complexity_value = float(score) if score else 0
                metrics["complexity"] = MetricResult(
                    name="Complexity Score",
                    value=float(complexity_value) if complexity_value else 0,
                    unit="score",
                    passed=True
                )

            # 3. Hardware fitness
            self._mcp_request_count += 1
            fitness_result = invoke_tool("calculate_hardware_fitness", qasm=qasm)
            if fitness_result.get("success"):
                score = fitness_result.get("score", {})
                if isinstance(score, dict):
                    fitness_value = score.get("fitness_score", score.get("fitness", 0))
                elif isinstance(score, list):
                    fitness_value = 0
                else:
                    fitness_value = float(score) if score else 0
                metrics["hardware_fitness"] = MetricResult(
                    name="Hardware Fitness",
                    value=float(fitness_value) if fitness_value else 0,
                    unit="score",
                    passed=fitness_value > 0.5 if fitness_value else False
                )

            # 4. Validation
            self._mcp_request_count += 1
            validation_result = invoke_tool("validate_syntax", qasm=qasm)
            valid_data = validation_result.get("valid", False)
            # Handle list or complex response
            if isinstance(valid_data, list):
                valid = "valid" in str(valid_data).lower() or "✅" in str(valid_data)
            elif isinstance(valid_data, dict):
                valid = valid_data.get("valid", False)
            else:
                valid = bool(valid_data) and validation_result.get("success", False)
            metrics["syntax_valid"] = MetricResult(
                name="Syntax Validation",
                value=1.0 if valid else 0.0,
                unit="boolean",
                passed=valid
            )

            # 5. Simulation correctness (if expected states defined)
            if problem.expected.expected_states:
                self._mcp_request_count += 1
                prob_result = invoke_tool("get_probabilities", qasm=qasm)
                if prob_result.get("success"):
                    probs = prob_result.get("probabilities", {})
                    if isinstance(probs, dict):
                        correctness = self._check_state_correctness(probs, problem.expected.expected_states)
                    else:
                        correctness = 0.5  # Default if can't parse
                    metrics["state_correctness"] = MetricResult(
                        name="State Correctness",
                        value=correctness,
                        unit="ratio",
                        passed=correctness > 0.9
                    )

        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")

        return metrics

    def _check_state_correctness(self, actual: Dict[str, float],
                                 expected: Dict[str, float]) -> float:
        """Check how close actual probabilities are to expected."""
        if not expected:
            return 1.0

        total_error = 0.0
        for state, expected_prob in expected.items():
            actual_prob = actual.get(state, 0.0)
            total_error += abs(expected_prob - actual_prob)

        # Normalize to 0-1 range (0 = perfect, 1 = worst)
        max_error = 2.0  # Maximum possible error
        correctness = 1.0 - (total_error / max_error)
        return max(0.0, correctness)

    def aggregate_results(self, results: List[EvaluationResult]) -> AggregatedResults:
        """Aggregate multiple run results with cost metrics."""
        if not results:
            return AggregatedResults(
                problem_id="",
                system_mode="",
                num_runs=0,
                success_rate=0.0,
                avg_time_ms=0.0,
                std_time_ms=0.0,
                avg_quality_score=0.0,
                effectiveness=0.0,
                reliability=0.0
            )

        problem_id = results[0].problem_id
        system_mode = results[0].system_mode
        num_runs = len(results)

        # Success rate
        successes = sum(1 for r in results if r.success)
        success_rate = successes / num_runs

        # Time statistics
        times = [r.execution_time_ms for r in results]
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0

        # Cost aggregates
        total_llm = sum(r.cost_metrics.llm_requests for r in results)
        total_mcp = sum(r.cost_metrics.mcp_requests for r in results)
        total_tokens = sum(r.cost_metrics.tokens_used for r in results)

        # Quality score (average of metric scores for successful runs)
        quality_scores = []
        cost_per_quality_scores = []
        for r in results:
            if r.success and r.metrics:
                # Combine relevant metrics
                scores = []
                if "complexity" in r.metrics:
                    # Invert complexity (lower is better)
                    scores.append(1.0 - min(r.metrics["complexity"].value / 100, 1.0))
                if "hardware_fitness" in r.metrics:
                    scores.append(r.metrics["hardware_fitness"].value)
                if "state_correctness" in r.metrics:
                    scores.append(r.metrics["state_correctness"].value)
                if scores:
                    q_score = statistics.mean(scores)
                    quality_scores.append(q_score)
                    cost_per_quality_scores.append(r.cost_metrics.cost_per_quality(q_score))

        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        avg_cpq = statistics.mean(cost_per_quality_scores) if cost_per_quality_scores else float('inf')

        # Effectiveness: Did we achieve the goal?
        effective_runs = sum(
            1 for r in results
            if r.success and r.metrics.get("state_correctness", MetricResult("", 0, "")).value > 0.8
        )
        effectiveness = effective_runs / num_runs if num_runs > 0 else 0.0

        # Reliability: Consistency of results (based on variance of success and quality)
        reliability = success_rate * (1.0 - std_time / max(avg_time, 1.0))
        reliability = max(0.0, min(1.0, reliability))

        return AggregatedResults(
            problem_id=problem_id,
            system_mode=system_mode,
            num_runs=num_runs,
            success_rate=success_rate,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            avg_quality_score=avg_quality,
            effectiveness=effectiveness,
            reliability=reliability,
            total_llm_requests=total_llm,
            total_mcp_requests=total_mcp,
            total_tokens=total_tokens,
            avg_cost_per_quality=avg_cpq,
            all_results=results
        )

    def evaluate_problem(self, problem: TestProblem,
                         modes: List[str] = None) -> Dict[str, AggregatedResults]:
        """Evaluate a problem across all modes."""
        if modes is None:
            modes = ["blackboard", "guided", "naked"]

        results_by_mode = {}

        for mode in modes:
            run_results = []

            for run_num in range(1, self.num_runs + 1):
                result = self.evaluate_single_run(problem, mode, run_num)
                run_results.append(result)

                # Store in database
                self.db.store_result(ResultEntry(
                    run_id=f"{problem.id}_{mode}_{run_num}",
                    system_mode=mode,
                    problem_id=problem.id,
                    success=result.success,
                    execution_time_ms=result.execution_time_ms,
                    circuit_qasm=result.circuit_qasm,
                    metrics={k: asdict(v) for k, v in result.metrics.items()}
                ))

            aggregated = self.aggregate_results(run_results)
            results_by_mode[mode] = aggregated

        return results_by_mode

    def evaluate_all(self, problems: List[TestProblem] = None,
                     modes: List[str] = None) -> Dict[str, Dict[str, AggregatedResults]]:
        """Evaluate all problems across all modes."""
        if problems is None:
            problems = ALL_PROBLEMS
        if modes is None:
            modes = ["blackboard", "guided", "naked"]

        all_results = {}

        for problem in problems:
            logger.info(f"Evaluating problem: {problem.name}")
            all_results[problem.id] = self.evaluate_problem(problem, modes)

        self.results = all_results
        return all_results

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a comparison report with cost analysis."""
        if not self.results:
            return "No results to report. Run evaluate_all() first."

        lines = [
            "=" * 100,
            "QUANTUM AGENT SYSTEM COMPARATIVE EVALUATION REPORT",
            f"Generated: {datetime.now().isoformat()}",
            f"Number of runs per problem: {self.num_runs}",
            "=" * 100,
            ""
        ]

        # Summary table with cost metrics
        lines.append("SUMMARY BY MODE (with Cost Analysis)")
        lines.append("-" * 100)
        lines.append(f"{'Mode':<12} {'Success%':>9} {'Time(ms)':>10} {'Quality':>8} {'LLM Req':>8} {'Tokens':>10} {'Cost/Qual':>10}")
        lines.append("-" * 100)

        mode_totals = {
            mode: {
                "success": 0, "total": 0, "times": [], "quality": [],
                "llm_req": 0, "mcp_req": 0, "tokens": 0, "cpq": []
            }
            for mode in ["blackboard", "guided", "naked"]
        }

        for problem_id, mode_results in self.results.items():
            for mode, agg in mode_results.items():
                mode_totals[mode]["success"] += agg.success_rate * agg.num_runs
                mode_totals[mode]["total"] += agg.num_runs
                mode_totals[mode]["times"].append(agg.avg_time_ms)
                mode_totals[mode]["quality"].append(agg.avg_quality_score)
                mode_totals[mode]["llm_req"] += agg.total_llm_requests
                mode_totals[mode]["mcp_req"] += agg.total_mcp_requests
                mode_totals[mode]["tokens"] += agg.total_tokens
                if agg.avg_cost_per_quality != float('inf'):
                    mode_totals[mode]["cpq"].append(agg.avg_cost_per_quality)

        for mode, totals in mode_totals.items():
            if totals["total"] > 0:
                success_pct = (totals["success"] / totals["total"]) * 100
                avg_time = statistics.mean(totals["times"]) if totals["times"] else 0
                avg_quality = statistics.mean(totals["quality"]) if totals["quality"] else 0
                avg_cpq = statistics.mean(totals["cpq"]) if totals["cpq"] else float('inf')
                cpq_str = f"{avg_cpq:.2f}" if avg_cpq != float('inf') else "N/A"

                lines.append(
                    f"{mode:<12} {success_pct:>8.1f}% {avg_time:>9.0f} {avg_quality:>8.2f} "
                    f"{totals['llm_req']:>8} {totals['tokens']:>10} {cpq_str:>10}"
                )

        lines.append("")
        lines.append("")

        # Cost efficiency analysis
        lines.append("COST EFFICIENCY ANALYSIS")
        lines.append("-" * 60)
        lines.append("")
        lines.append("Expected LLM Requests per problem:")
        lines.append("  - Naked:      1 (single direct LLM call)")
        lines.append("  - Guided:     4 (one per agent: Architect, Builder, Validator, Scorer)")
        lines.append("  - Blackboard: 8-12 (multiple collaborative rounds)")
        lines.append("")
        lines.append("Cost-per-Quality interpretation:")
        lines.append("  - Lower is better (less resources for same quality)")
        lines.append("  - Naked has lowest cost but tests raw LLM capability")
        lines.append("  - Blackboard has highest cost but best quality potential")
        lines.append("")

        # Detailed results per problem
        lines.append("DETAILED RESULTS BY PROBLEM")
        lines.append("-" * 100)

        for problem_id, mode_results in self.results.items():
            problem = get_problem(problem_id)
            problem_name = problem.name if problem else problem_id

            lines.append(f"\n{problem_name} ({problem_id})")
            lines.append("-" * 50)
            lines.append(f"{'Mode':<12} {'Success':>8} {'Time(ms)':>10} {'Quality':>8} {'LLM':>6} {'Tokens':>8}")

            for mode, agg in mode_results.items():
                lines.append(
                    f"{mode:<12} "
                    f"{agg.success_rate*100:>7.0f}% "
                    f"{agg.avg_time_ms:>9.0f} "
                    f"{agg.avg_quality_score:>8.2f} "
                    f"{agg.total_llm_requests:>6} "
                    f"{agg.total_tokens:>8}"
                )

        lines.append("")
        lines.append("=" * 100)
        lines.append("END OF REPORT")

        report = "\n".join(lines)

        if output_path:
            output_path.write_text(report)
            logger.info(f"Report saved to: {output_path}")

        return report

    def export_csv(self, output_path: Optional[Path] = None) -> str:
        """
        Export results to CSV for research analysis.
        
        CSV Columns:
        - timestamp: When the evaluation was run
        - problem_id: Unique problem identifier
        - problem_name: Human-readable problem name
        - difficulty: Problem difficulty (easy, medium, hard)
        - mode: Execution mode (naked, guided, blackboard)
        - run_number: Run iteration (1 to num_runs)
        - success: Whether the run succeeded (True/False)
        - time_ms: Execution time in milliseconds
        - llm_requests: Number of LLM API calls
        - tokens_used: Total tokens consumed
        - mcp_requests: Number of MCP tool calls
        - quality_score: Combined quality score (0-1)
        - depth: Circuit depth
        - complexity: Circuit complexity score
        - hardware_fitness: Hardware compatibility score
        - syntax_valid: Whether QASM syntax is valid
        - state_correctness: Probability distribution correctness
        - cost_per_quality: Cost efficiency ratio
        - model_used: Primary LLM model used
        - qasm_length: Length of generated QASM code
        """
        if not self.results:
            return "No results to export. Run evaluate_all() first."

        timestamp = datetime.now().isoformat()
        
        # Default output path
        if output_path is None:
            output_dir = Path(__file__).parent.parent / "research"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # CSV header
        fieldnames = [
            'timestamp', 'problem_id', 'problem_name', 'difficulty',
            'mode', 'run_number', 'success', 'time_ms',
            'llm_requests', 'tokens_used', 'mcp_requests',
            'quality_score', 'depth', 'complexity', 'hardware_fitness',
            'syntax_valid', 'state_correctness', 'cost_per_quality',
            'model_used', 'qasm_length', 'errors'
        ]

        rows = []
        
        for problem_id, mode_results in self.results.items():
            problem = get_problem(problem_id)
            problem_name = problem.name if problem else problem_id
            difficulty = problem.difficulty if problem else "unknown"
            
            for mode, agg in mode_results.items():
                for result in agg.all_results:
                    # Extract metric values safely
                    def get_metric(name, default=0.0):
                        if name in result.metrics:
                            return result.metrics[name].value
                        return default
                    
                    # Calculate quality score
                    quality_components = []
                    if "complexity" in result.metrics:
                        quality_components.append(1.0 - min(get_metric("complexity") / 100, 1.0))
                    if "hardware_fitness" in result.metrics:
                        quality_components.append(get_metric("hardware_fitness"))
                    if "state_correctness" in result.metrics:
                        quality_components.append(get_metric("state_correctness"))
                    quality_score = statistics.mean(quality_components) if quality_components else 0.0
                    
                    # Cost per quality
                    cpq = result.cost_metrics.cost_per_quality(quality_score) if quality_score > 0 else float('inf')
                    cpq_str = f"{cpq:.4f}" if cpq != float('inf') else "inf"
                    
                    # Model used
                    models = result.cost_metrics.models_used
                    model_used = models[0] if models else "unknown"
                    
                    # QASM length
                    qasm_len = len(result.circuit_qasm) if result.circuit_qasm else 0
                    
                    row = {
                        'timestamp': timestamp,
                        'problem_id': problem_id,
                        'problem_name': problem_name,
                        'difficulty': difficulty,
                        'mode': mode,
                        'run_number': result.run_number,
                        'success': result.success,
                        'time_ms': f"{result.execution_time_ms:.2f}",
                        'llm_requests': result.cost_metrics.llm_requests,
                        'tokens_used': result.cost_metrics.tokens_used,
                        'mcp_requests': result.cost_metrics.mcp_requests,
                        'quality_score': f"{quality_score:.4f}",
                        'depth': get_metric("depth"),
                        'complexity': f"{get_metric('complexity'):.2f}",
                        'hardware_fitness': f"{get_metric('hardware_fitness'):.4f}",
                        'syntax_valid': get_metric("syntax_valid") == 1.0,
                        'state_correctness': f"{get_metric('state_correctness'):.4f}",
                        'cost_per_quality': cpq_str,
                        'model_used': model_used,
                        'qasm_length': qasm_len,
                        'errors': "; ".join(result.errors) if result.errors else ""
                    }
                    rows.append(row)

        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"CSV exported to: {output_path}")
        return str(output_path)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the evaluation run.
        Useful for programmatic access to results.
        """
        if not self.results:
            return {}

        stats = {
            'timestamp': datetime.now().isoformat(),
            'num_problems': len(self.results),
            'runs_per_problem': self.num_runs,
            'modes': {}
        }

        for mode in ['naked', 'guided', 'blackboard']:
            mode_stats = {
                'success_rate': 0.0,
                'avg_time_ms': 0.0,
                'total_llm_requests': 0,
                'total_tokens': 0,
                'avg_quality': 0.0
            }
            
            times = []
            qualities = []
            total_runs = 0
            successes = 0
            
            for problem_id, mode_results in self.results.items():
                if mode in mode_results:
                    agg = mode_results[mode]
                    total_runs += agg.num_runs
                    successes += agg.success_rate * agg.num_runs
                    times.append(agg.avg_time_ms)
                    qualities.append(agg.avg_quality_score)
                    mode_stats['total_llm_requests'] += agg.total_llm_requests
                    mode_stats['total_tokens'] += agg.total_tokens
            
            if total_runs > 0:
                mode_stats['success_rate'] = successes / total_runs
                mode_stats['avg_time_ms'] = statistics.mean(times) if times else 0
                mode_stats['avg_quality'] = statistics.mean(qualities) if qualities else 0
            
            stats['modes'][mode] = mode_stats

        return stats
