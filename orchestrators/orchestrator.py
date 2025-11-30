# Path: QAgents-workflos/orchestrators/orchestrator.py
# Relations: Uses agents, workflows, database modules
# Description: Orchestrators for Blackboard, Guided, and Naked execution modes
"""
Orchestrators Module: Workflow orchestration and execution.
Contains both Blackboard (free) and Guided (strict) orchestrators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import time

from agents import (
    BaseAgent, AgentContext, AgentResult,
    AgentState, create_all_agents
)
from workflows import (
    WorkflowDefinition, WorkflowExecution,
    WorkflowStatus, get_workflow
)
from database import get_database, LogEntry

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result from orchestrator execution."""
    success: bool
    final_output: Any
    execution_time_ms: float
    steps_completed: int
    total_steps: int
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class BaseOrchestrator(ABC):
    """Abstract base class for orchestrators."""

    def __init__(self, name: str):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.db = get_database()

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent

    def log(self, level: str, message: str, workflow_id: str = None, data: Dict = None):
        """Log orchestrator activity."""
        entry = LogEntry(
            level=level,
            message=message,
            agent_id=self.name,
            workflow_id=workflow_id,
            data=data or {}
        )
        self.db.log(entry)

    @abstractmethod
    def run(self, goal: str, initial_context: Dict = None) -> OrchestratorResult:
        """Run the orchestrator to achieve the goal."""
        pass


class BlackboardOrchestrator(BaseOrchestrator):
    """
    Blackboard (Free) Orchestrator.

    Uses a shared blackboard for agent communication.
    Agents opportunistically activate when they can contribute.
    Emergent workflow based on data availability.
    """

    def __init__(self):
        super().__init__("blackboard")
        self.blackboard: Dict[str, Any] = {}
        self.max_iterations = 20

    def _reset_blackboard(self, goal: str, initial_context: Dict = None):
        """Initialize the blackboard with goal and context."""
        # Ensure goal is a string
        if isinstance(goal, list):
            goal = goal[0] if goal else ""
        goal = str(goal) if goal else ""
        
        self.blackboard = {
            "goal": goal,
            "current_circuit": None,
            "validation_passed": False,
            "scores": None,
            "completed": False,
            **(initial_context or {})
        }

    def _build_context(self) -> AgentContext:
        """Build agent context from blackboard."""
        return AgentContext(
            goal=self.blackboard.get("goal", ""),
            current_circuit=self.blackboard.get("current_circuit"),
            history=self.blackboard.get("history", []),
            constraints=self.blackboard.get("constraints", {}),
            shared_data=self.blackboard
        )

    def _find_active_agent(self, context: AgentContext) -> Optional[BaseAgent]:
        """Find an agent that can handle the current state."""
        # Priority order for agent selection - simplified for reliability
        # First: generate circuit, then validate
        priority_order = ["builder", "architect", "validator"]

        for agent_id in priority_order:
            agent = self.agents.get(agent_id)
            if agent and agent.can_handle(context):
                if agent.state == AgentState.IDLE:
                    return agent

        return None

    def _update_blackboard(self, agent_id: str, result: AgentResult):
        """Update blackboard with agent results."""
        if not result.success:
            return

        data = result.data
        if isinstance(data, dict):
            # Extract QASM if present
            if "qasm" in data:
                qasm = data["qasm"]
                # Handle list responses
                if isinstance(qasm, list):
                    qasm = qasm[0] if qasm else None
                self.blackboard["current_circuit"] = qasm

            # Update validation status
            if "valid" in data:
                self.blackboard["validation_passed"] = data["valid"]

            # Update scores
            if "score" in data:
                self.blackboard["scores"] = data["score"]

        # Track history
        if "history" not in self.blackboard:
            self.blackboard["history"] = []
        self.blackboard["history"].append({
            "agent": agent_id,
            "action": result.actions_taken,
            "success": result.success,
            "timestamp": datetime.now().isoformat()
        })

    def _check_completion(self) -> bool:
        """Check if the goal has been achieved."""
        # Simple completion: we have a validated circuit
        has_circuit = self.blackboard.get("current_circuit") is not None
        is_validated = self.blackboard.get("validation_passed", False)
        return has_circuit and is_validated

    def run(self, goal: str, initial_context: Dict = None) -> OrchestratorResult:
        """Run blackboard orchestration."""
        start_time = time.perf_counter()

        self.log("INFO", f"Starting blackboard orchestration for: {goal}")
        self._reset_blackboard(goal, initial_context)

        # Ensure we have agents
        if not self.agents:
            self.agents = create_all_agents()

        agent_results = {}
        steps_completed = 0
        errors = []

        for iteration in range(self.max_iterations):
            context = self._build_context()

            # Find an agent that can work
            agent = self._find_active_agent(context)

            if agent is None:
                self.log("INFO", "No active agent found, checking completion")
                if self._check_completion():
                    break
                # No agent and not complete - might be stuck
                if iteration > 5:  # Give it a few tries
                    errors.append("No agent could make progress")
                    break
                continue

            self.log("INFO", f"Activating agent: {agent.agent_id}")

            # Agent decides and executes - with null safety
            try:
                action = agent.decide(context)
                if action is None:
                    self.log("WARN", f"Agent {agent.agent_id} returned no action, continuing")
                    agent.reset()
                    continue

                result = agent.execute(action, context)
                if result is None:
                    self.log("WARN", f"Agent {agent.agent_id} returned no result, continuing")
                    agent.reset()
                    continue
                    
                agent_results[agent.agent_id] = result
                steps_completed += 1

                # Update blackboard
                self._update_blackboard(agent.agent_id, result)
                
            except Exception as e:
                self.log("ERROR", f"Agent {agent.agent_id} failed: {e}")
                errors.append(f"Agent {agent.agent_id} error: {str(e)}")
                agent.reset()
                continue

            # Reset agent for next potential activation
            agent.reset()

            # Check completion
            if self._check_completion():
                self.log("INFO", "Goal achieved!")
                break

        elapsed = (time.perf_counter() - start_time) * 1000

        return OrchestratorResult(
            success=self._check_completion(),
            final_output=self.blackboard.get("current_circuit"),
            execution_time_ms=elapsed,
            steps_completed=steps_completed,
            total_steps=self.max_iterations,
            agent_results=agent_results,
            errors=errors
        )


class GuidedOrchestrator(BaseOrchestrator):
    """
    Guided (Strict) Orchestrator.

    Follows a predefined workflow with explicit steps.
    Central control over agent execution order.
    Predictable, auditable execution path.
    """

    def __init__(self, workflow_name: str = "build"):
        super().__init__("guided")
        self.workflow = get_workflow(workflow_name)
        if self.workflow is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        self.execution: Optional[WorkflowExecution] = None

    def set_workflow(self, workflow_name: str):
        """Change the workflow."""
        self.workflow = get_workflow(workflow_name)
        if self.workflow is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")

    def run(self, goal: str, initial_context: Dict = None) -> OrchestratorResult:
        """Run guided workflow orchestration."""
        start_time = time.perf_counter()

        # Ensure goal is a string
        if isinstance(goal, list):
            goal = goal[0] if goal else ""
        goal = str(goal) if goal else ""

        self.log("INFO", f"Starting guided workflow '{self.workflow.name}' for: {goal}")

        # Initialize execution state
        self.execution = WorkflowExecution(
            workflow=self.workflow,
            context={"goal": goal, **(initial_context or {})}
        )
        self.execution.status = WorkflowStatus.IN_PROGRESS

        # Ensure we have agents
        if not self.agents:
            self.agents = create_all_agents()

        agent_results = {}

        # Execute each step in order
        while self.execution.current_step is not None:
            step = self.execution.current_step
            self.log("INFO", f"Executing step: {step.name} ({step.agent_type})")

            # Get the agent for this step
            agent = self.agents.get(step.agent_type)
            if agent is None:
                if step.required:
                    self.execution.fail(f"Missing agent: {step.agent_type}")
                    break
                else:
                    self.log("WARN", f"Skipping optional step: {step.name}")
                    self.execution.advance()
                    continue

            # Build context for agent
            context = AgentContext(
                goal=self.execution.context.get("goal", ""),
                current_circuit=self.execution.context.get("circuit_qasm"),
                history=[],
                constraints={},
                shared_data=self.execution.context
            )

            # Agent decides and executes
            action = agent.decide(context)
            if action is None:
                # Agent has nothing to do - might be okay for some steps
                self.log("WARN", f"Agent {step.agent_type} returned no action")
                self.execution.advance()
                continue

            result = agent.execute(action, context)
            agent_results[step.name] = result

            # Store outputs in execution context
            if result.success and result.data:
                for output_key in step.outputs:
                    if isinstance(result.data, dict):
                        if output_key in result.data:
                            self.execution.context[output_key] = result.data[output_key]
                        elif "qasm" in result.data:
                            qasm = result.data["qasm"]
                            # Handle list responses
                            if isinstance(qasm, list):
                                qasm = qasm[0] if qasm else None
                            self.execution.context["circuit_qasm"] = qasm

            # Handle failure
            if not result.success and step.required:
                self.execution.fail(f"Step {step.name} failed: {result.message}")
                break

            # Reset agent and advance
            agent.reset()
            self.execution.advance()

        elapsed = (time.perf_counter() - start_time) * 1000

        return OrchestratorResult(
            success=self.execution.status == WorkflowStatus.COMPLETED,
            final_output=self.execution.context.get(self.workflow.final_output),
            execution_time_ms=elapsed,
            steps_completed=self.execution.current_step_index,
            total_steps=len(self.workflow.steps),
            agent_results=agent_results,
            errors=self.execution.errors
        )


class NakedOrchestrator(BaseOrchestrator):
    """
    Naked (Baseline) Orchestrator.

    Direct LLM-to-QASM generation with single call.
    No multi-agent coordination, no structured workflow.
    Uses ONE LLM call per problem for baseline comparison.
    
    Purpose: Measure raw LLM capability at quantum circuit generation
    without agentic overhead.
    """

    def __init__(self):
        super().__init__("naked")
        self._llm = None

    def _get_llm(self):
        """Lazy load LLM adapter."""
        if self._llm is None:
            from agents.llm_adapter import get_llm_adapter
            from config import config
            self._llm = get_llm_adapter(
                provider="gemini",
                api_key=config.llm.api_key,
                enable_fallback=True
            )
        return self._llm

    def run(self, goal: str, initial_context: Dict = None) -> OrchestratorResult:
        """
        Run naked LLM execution - ONE LLM call per problem.
        
        This is the baseline test: can a single LLM call generate
        valid QASM for a quantum computing problem?
        """
        start_time = time.perf_counter()

        # Ensure goal is a string
        if isinstance(goal, list):
            goal = goal[0] if goal else ""
        goal = str(goal) if goal else ""

        self.log("INFO", f"Starting naked LLM execution for: {goal}")

        from tools import invoke_tool

        errors = []
        circuit_qasm = None
        llm_requests = 0
        tokens_used = 0

        # System prompt for direct QASM generation
        system_prompt = """You are an expert quantum computing engineer.
Your task is to generate valid OpenQASM 2.0 code for the given quantum circuit problem.

RULES:
1. Output ONLY valid OpenQASM 2.0 code
2. Start with: OPENQASM 2.0; include "qelib1.inc";
3. Declare qubits with: qreg q[N];
4. Declare classical bits with: creg c[N];
5. Use standard gates: h, x, y, z, cx, cz, ccx, swap, t, s, rx, ry, rz
6. Add measurements with: measure q[i] -> c[i];
7. NO explanations, NO markdown, ONLY QASM code

EXAMPLE OUTPUT:
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

        user_prompt = f"""Generate the OpenQASM 2.0 code for this quantum circuit problem:

{goal}

Output ONLY the QASM code, nothing else."""

        try:
            # Single LLM call - the naked baseline test
            llm = self._get_llm()
            response = llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for deterministic output
                max_tokens=1000
            )
            llm_requests = 1
            tokens_used = response.tokens_used

            # Extract QASM from response
            raw_output = response.text.strip()
            
            # Clean up common LLM artifacts
            if "```" in raw_output:
                # Extract from code block
                lines = raw_output.split("\n")
                in_block = False
                qasm_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_block:
                            break
                        in_block = True
                        continue
                    if in_block:
                        qasm_lines.append(line)
                raw_output = "\n".join(qasm_lines)
            
            # Ensure it starts with OPENQASM declaration
            if "OPENQASM" in raw_output:
                # Find the start of QASM
                idx = raw_output.find("OPENQASM")
                circuit_qasm = raw_output[idx:]
            else:
                # Try to use as-is if it looks like QASM
                if "qreg" in raw_output or "include" in raw_output:
                    circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" + raw_output
                else:
                    errors.append(f"LLM did not produce valid QASM: {raw_output[:100]}")

            # Validate the generated QASM
            if circuit_qasm:
                validation = invoke_tool("validate_syntax", qasm=circuit_qasm)
                if not validation.get("success") or not validation.get("valid", False):
                    error_msg = validation.get("error", "Unknown validation error")
                    errors.append(f"QASM validation failed: {error_msg}")
                    # Still keep the circuit for analysis
                    self.log("WARN", f"Generated QASM failed validation: {error_msg}")

        except Exception as e:
            errors.append(str(e))
            self.log("ERROR", f"Naked LLM execution failed: {e}")

        elapsed = (time.perf_counter() - start_time) * 1000

        # Create a simple AgentResult-like dict for compatibility
        from agents import AgentResult
        naked_result = AgentResult(
            success=circuit_qasm is not None and len(errors) == 0,
            data={
                "qasm": circuit_qasm,
                "llm_requests": llm_requests,
                "tokens_used": tokens_used
            },
            message=f"Generated QASM via naked LLM ({llm_requests} request, {tokens_used} tokens)"
        )

        return OrchestratorResult(
            success=circuit_qasm is not None and len(errors) == 0,
            final_output=circuit_qasm,
            execution_time_ms=elapsed,
            steps_completed=1 if llm_requests > 0 else 0,
            total_steps=1,
            agent_results={"naked_llm": naked_result},
            errors=errors
        )


# Factory function
def create_orchestrator(mode: str) -> BaseOrchestrator:
    """Create an orchestrator based on mode."""
    if mode == "blackboard":
        return BlackboardOrchestrator()
    elif mode == "guided":
        return GuidedOrchestrator()
    elif mode == "naked":
        return NakedOrchestrator()
    elif mode == "quasar":
        from .quasar_orchestrator import QuasarOrchestrator
        return QuasarOrchestrator()
    elif mode == "hybrid":
        from .quasar_orchestrator import HybridOrchestrator
        return HybridOrchestrator()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'blackboard', 'guided', 'naked', 'quasar', or 'hybrid'")
