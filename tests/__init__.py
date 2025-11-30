"""Tests module: Test problems and evaluation harness."""

from .test_problems import (
    ProblemDifficulty,
    ProblemCategory,
    ExpectedOutput,
    TestProblem,
    # Problems by ID naming
    PROBLEM_E1_PHASE_FLIP,
    PROBLEM_E2_CONTROLLED_NOT,
    PROBLEM_E3_MEASUREMENT_BASIS,
    PROBLEM_M1_SWAP_DECOMPOSITION,
    PROBLEM_M2_CONTROLLED_Z,
    PROBLEM_M3_PHASE_ESTIMATION_PREP,
    PROBLEM_H1_DEUTSCH,
    PROBLEM_H2_GROVER_2QUBIT,
    PROBLEM_H3_TELEPORTATION_PREP,
    # Collections
    EASY_PROBLEMS,
    MEDIUM_PROBLEMS,
    HARD_PROBLEMS,
    ALL_PROBLEMS,
    get_problem,
    get_problems_by_difficulty,
    get_problems_by_category,
    get_problems_by_tag,
    get_research_problem_set
)

from .evaluation_harness import (
    MetricResult,
    CostMetrics,
    EvaluationResult,
    AggregatedResults,
    EvaluationHarness
)

from .circuit_quality_analyzer import (
    CircuitQualityAnalyzer,
    AnalysisResult,
    get_analyzer
)

from .quality_evaluation_harness import (
    QualityEvaluationHarness,
    run_quick_quality_test
)

# Backward compatibility aliases
BELL_STATE_PROBLEM = PROBLEM_E2_CONTROLLED_NOT  # Bell state is easy_002

__all__ = [
    "ProblemDifficulty",
    "ProblemCategory",
    "ExpectedOutput",
    "TestProblem",
    "PROBLEM_E1_PHASE_FLIP",
    "PROBLEM_E2_CONTROLLED_NOT",
    "PROBLEM_E3_MEASUREMENT_BASIS",
    "PROBLEM_M1_SWAP_DECOMPOSITION",
    "PROBLEM_M2_CONTROLLED_Z",
    "PROBLEM_M3_PHASE_ESTIMATION_PREP",
    "PROBLEM_H1_DEUTSCH",
    "PROBLEM_H2_GROVER_2QUBIT",
    "PROBLEM_H3_TELEPORTATION_PREP",
    "EASY_PROBLEMS",
    "MEDIUM_PROBLEMS",
    "HARD_PROBLEMS",
    "ALL_PROBLEMS",
    "get_problem",
    "get_problems_by_difficulty",
    "get_problems_by_category",
    "get_problems_by_tag",
    "get_research_problem_set",
    "MetricResult",
    "CostMetrics",
    "EvaluationResult",
    "AggregatedResults",
    "EvaluationHarness",
    "BELL_STATE_PROBLEM",
    # Quality analysis
    "CircuitQualityAnalyzer",
    "AnalysisResult",
    "get_analyzer",
    "QualityEvaluationHarness",
    "run_quick_quality_test"
]