"""
Workflows Module: Predefined workflow definitions.
Workflows are sequences of steps that produce useful outputs.

EXPECTED REQUEST COUNTS PER WORKFLOW TYPE:
==========================================

NAKED (Baseline - Direct MCP):
  - LLM requests: 0 per problem
  - MCP requests: 1-2 per problem (direct circuit generation)
  - Total API calls: 1-2 per problem
  - Rate limit impact: NONE (no LLM calls)
  - Expected time: <1 second per problem

GUIDED (Rigid Agentic - Rule-Based State Machine):
  - LLM requests: 4 per problem (one per agent: Architect, Builder, Validator, Scorer)
  - MCP requests: 2-4 per problem (template selection, circuit generation)
  - Total API calls: 6-8 per problem
  - Rate limit impact: LOW (sequential agent calls with 5s rate limiting)
  - Expected time: ~20-30 seconds per problem with rate limiting

BLACKBOARD (Flexible Agentic - Event-Driven):
  - LLM requests: 8-12 per problem (multiple collaborative rounds)
  - MCP requests: 4-8 per problem (iterative refinement)
  - Total API calls: 12-20 per problem
  - Rate limit impact: MODERATE (many LLM calls, needs careful rate management)
  - Expected time: ~60-90 seconds per problem with rate limiting

For 9 test problems (3 easy, 3 medium, 3 hard):
  - Naked: ~9-18 API calls total (all MCP, no rate limiting) = ~9 seconds
  - Guided: ~54-72 API calls (36 LLM + 18-36 MCP) = ~3-6 minutes with rate limiting
  - Blackboard: ~108-180 API calls (72-108 LLM + 36-72 MCP) = ~6-15 minutes

Free tier limits (Gemini 2.5 Flash-Lite): 15 RPM, 1000 RPD
With 80% buffer (12 RPM = 5s intervals): Can process ~2-3 Guided problems/min or ~1 Blackboard problem/min
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    agent_type: str
    description: str
    required: bool = True
    timeout_seconds: float = 60.0
    retry_count: int = 1
    inputs: List[str] = field(default_factory=list)  # Keys from context
    outputs: List[str] = field(default_factory=list)  # Keys to store in context


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow."""
    name: str
    description: str
    steps: List[WorkflowStep]
    entry_point: str = ""  # First step name
    final_output: str = ""  # Key for final result
    
    def __post_init__(self):
        if not self.entry_point and self.steps:
            self.entry_point = self.steps[0].name


@dataclass
class WorkflowExecution:
    """Runtime state of workflow execution."""
    workflow: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    current_step_index: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    @property
    def current_step(self) -> Optional[WorkflowStep]:
        if 0 <= self.current_step_index < len(self.workflow.steps):
            return self.workflow.steps[self.current_step_index]
        return None
        
    def advance(self):
        """Move to next step."""
        self.current_step_index += 1
        if self.current_step_index >= len(self.workflow.steps):
            self.status = WorkflowStatus.COMPLETED
            
    def fail(self, error: str):
        """Mark workflow as failed."""
        self.errors.append(error)
        self.status = WorkflowStatus.FAILED


# ============================================================
# PREDEFINED WORKFLOWS
# ============================================================

BUILD_WORKFLOW = WorkflowDefinition(
    name="build",
    description="Create a new quantum circuit from a description or template",
    steps=[
        WorkflowStep(
            name="plan",
            agent_type="architect",
            description="Plan the circuit structure",
            inputs=["goal"],
            outputs=["plan", "circuit_qasm"]
        ),
        WorkflowStep(
            name="build",
            agent_type="builder",
            description="Build the circuit based on plan",
            inputs=["plan"],
            outputs=["circuit_qasm"]
        ),
        WorkflowStep(
            name="validate",
            agent_type="validator",
            description="Validate the built circuit",
            inputs=["circuit_qasm"],
            outputs=["validation_result"]
        ),
        WorkflowStep(
            name="score",
            agent_type="scorer",
            description="Score the final circuit",
            inputs=["circuit_qasm"],
            outputs=["scores"],
            required=False
        )
    ],
    final_output="circuit_qasm"
)


OPTIMIZE_WORKFLOW = WorkflowDefinition(
    name="optimize",
    description="Optimize an existing quantum circuit",
    steps=[
        WorkflowStep(
            name="analyze",
            agent_type="analyzer",
            description="Analyze the current circuit",
            inputs=["circuit_qasm"],
            outputs=["analysis"]
        ),
        WorkflowStep(
            name="optimize",
            agent_type="optimizer",
            description="Apply optimizations",
            inputs=["circuit_qasm", "analysis"],
            outputs=["optimized_qasm"]
        ),
        WorkflowStep(
            name="validate",
            agent_type="validator",
            description="Validate optimized circuit",
            inputs=["optimized_qasm"],
            outputs=["validation_result"]
        ),
        WorkflowStep(
            name="compare",
            agent_type="scorer",
            description="Compare before/after scores",
            inputs=["circuit_qasm", "optimized_qasm"],
            outputs=["comparison"]
        )
    ],
    final_output="optimized_qasm"
)


EVALUATE_WORKFLOW = WorkflowDefinition(
    name="evaluate",
    description="Evaluate a quantum circuit comprehensively",
    steps=[
        WorkflowStep(
            name="validate",
            agent_type="validator",
            description="Validate circuit correctness",
            inputs=["circuit_qasm"],
            outputs=["validation_result"]
        ),
        WorkflowStep(
            name="analyze",
            agent_type="analyzer",
            description="Analyze circuit properties",
            inputs=["circuit_qasm"],
            outputs=["analysis"]
        ),
        WorkflowStep(
            name="score",
            agent_type="scorer",
            description="Score the circuit",
            inputs=["circuit_qasm"],
            outputs=["scores"]
        ),
        WorkflowStep(
            name="simulate",
            agent_type="simulator",
            description="Simulate and get results",
            inputs=["circuit_qasm"],
            outputs=["simulation_results"]
        )
    ],
    final_output="scores"
)


FULL_PIPELINE_WORKFLOW = WorkflowDefinition(
    name="full_pipeline",
    description="Complete circuit creation, optimization, and evaluation",
    steps=[
        WorkflowStep(
            name="plan",
            agent_type="architect",
            description="Plan circuit architecture",
            inputs=["goal"],
            outputs=["plan"]
        ),
        WorkflowStep(
            name="build",
            agent_type="builder",
            description="Build initial circuit",
            inputs=["plan"],
            outputs=["circuit_qasm"]
        ),
        WorkflowStep(
            name="validate_initial",
            agent_type="validator",
            description="Validate initial build",
            inputs=["circuit_qasm"],
            outputs=["initial_validation"]
        ),
        WorkflowStep(
            name="analyze",
            agent_type="analyzer",
            description="Analyze for optimization",
            inputs=["circuit_qasm"],
            outputs=["analysis"]
        ),
        WorkflowStep(
            name="optimize",
            agent_type="optimizer",
            description="Optimize circuit",
            inputs=["circuit_qasm", "analysis"],
            outputs=["optimized_qasm"],
            required=False
        ),
        WorkflowStep(
            name="validate_final",
            agent_type="validator",
            description="Validate final circuit",
            inputs=["optimized_qasm"],
            outputs=["final_validation"]
        ),
        WorkflowStep(
            name="score",
            agent_type="scorer",
            description="Final scoring",
            inputs=["optimized_qasm"],
            outputs=["scores"]
        )
    ],
    final_output="optimized_qasm"
)


# Registry of available workflows
WORKFLOWS = {
    "build": BUILD_WORKFLOW,
    "optimize": OPTIMIZE_WORKFLOW,
    "evaluate": EVALUATE_WORKFLOW,
    "full_pipeline": FULL_PIPELINE_WORKFLOW
}


def get_workflow(name: str) -> Optional[WorkflowDefinition]:
    """Get a workflow by name."""
    return WORKFLOWS.get(name)


def list_workflows() -> List[str]:
    """List all available workflow names."""
    return list(WORKFLOWS.keys())
