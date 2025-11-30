# Path: QAgents-workflos/agents/specialized_agents.py
# Relations: Uses base_agent.py, prompts/agent_prompts.py
# Description: Domain-specific agents for quantum circuit optimization
"""
Specialized Quantum Agents: Domain-specific agents for circuit optimization.
"""

from typing import Optional, List, Dict, Any
from .base_agent import (
    LLMAgent, RuleBasedAgent, AgentRole,
    AgentContext, AgentAction, AgentResult
)


def _goal_to_string(context: AgentContext) -> str:
    """Safely extract goal as string from context."""
    goal = context.goal
    if isinstance(goal, list):
        goal = goal[0] if goal else ""
    return str(goal).lower() if goal else ""


class ArchitectAgent(LLMAgent):
    """
    Plans the overall circuit structure.
    Decides what type of circuit to build and the high-level approach.
    """

    def __init__(self, agent_id: str = "architect"):
        from prompts import ARCHITECT_PROMPT

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ARCHITECT,
            system_prompt=ARCHITECT_PROMPT,
            tools=[
                "create_from_template",
                "generate_from_description",
                "analyze_circuit"
            ]
        )

    def can_handle(self, context: AgentContext) -> bool:
        """Can handle when no circuit exists or replanning needed."""
        goal = _goal_to_string(context)
        return context.current_circuit is None or "replan" in goal


class BuilderAgent(LLMAgent):
    """
    Builds and modifies circuits based on plans.
    Handles the actual circuit construction.
    """

    def __init__(self, agent_id: str = "builder"):
        from prompts import BUILDER_PROMPT

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.BUILDER,
            system_prompt=BUILDER_PROMPT,
            tools=[
                "create_from_template",
                "generate_random_circuit",
                "generate_from_description",
                "compose_circuits",
                "tensor_circuits",
                "repeat_circuit"
            ]
        )

    def can_handle(self, context: AgentContext) -> bool:
        """Can handle when we need to build a circuit."""
        has_plan = any("plan" in str(h.get("action", "")).lower() for h in context.history)
        no_circuit = context.current_circuit is None
        return has_plan or no_circuit


class ValidatorAgent(LLMAgent):
    """
    Validates circuits for correctness and hardware compatibility.
    """

    def __init__(self, agent_id: str = "validator"):
        from prompts import VALIDATOR_PROMPT

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.VALIDATOR,
            system_prompt=VALIDATOR_PROMPT,
            tools=[
                "validate_syntax",
                "check_connectivity",
                "verify_unitary"
            ]
        )

    def can_handle(self, context: AgentContext) -> bool:
        """Can handle when there's a circuit to validate."""
        return context.current_circuit is not None


class OptimizerAgent(LLMAgent):
    """
    Optimizes circuits for depth, gate count, and hardware fitness.
    """

    def __init__(self, agent_id: str = "optimizer"):
        from prompts import OPTIMIZER_PROMPT

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.OPTIMIZER,
            system_prompt=OPTIMIZER_PROMPT,
            tools=[
                "generate_inverse",
                "compose_circuits",
                "analyze_circuit",
                "calculate_complexity",
                "calculate_hardware_fitness"
            ]
        )

    def can_handle(self, context: AgentContext) -> bool:
        """Can handle when circuit exists and optimization is needed."""
        if context.current_circuit is None:
            return False
        goal = _goal_to_string(context)
        return "optimize" in goal or "improve" in goal


class AnalyzerAgent(LLMAgent):
    """
    Analyzes circuit properties and provides insights.
    """

    def __init__(self, agent_id: str = "analyzer"):
        from prompts import ANALYZER_PROMPT

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ANALYZER,
            system_prompt=ANALYZER_PROMPT,
            tools=[
                "parse_qasm",
                "analyze_circuit",
                "get_circuit_depth",
                "get_statevector",
                "get_probabilities",
                "estimate_resources",
                "estimate_noise"
            ]
        )
        
    def can_handle(self, context: AgentContext) -> bool:
        """Can handle when circuit exists and analysis is needed."""
        return context.current_circuit is not None


class ScorerAgent(LLMAgent):
    """
    Scores circuits on various metrics.
    """

    def __init__(self, agent_id: str = "scorer"):
        from prompts import SCORER_PROMPT

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.SCORER,
            system_prompt=SCORER_PROMPT,
            tools=[
                "calculate_complexity",
                "calculate_hardware_fitness",
                "calculate_expressibility",
                "simulate_circuit"
            ]
        )

    def can_handle(self, context: AgentContext) -> bool:
        """Can handle when circuit exists and scoring is requested."""
        if context.current_circuit is None:
            return False
        goal = _goal_to_string(context)
        return "score" in goal or "evaluate" in goal


class SimulatorAgent(RuleBasedAgent):
    """
    Rule-based agent for circuit simulation.
    Deterministic - always simulates when circuit is ready.
    """

    def __init__(self, agent_id: str = "simulator"):
        def simulate_rule(context: AgentContext) -> Optional[AgentAction]:
            if context.current_circuit:
                return AgentAction(
                    tool_name="simulate_circuit",
                    arguments={"qasm": context.current_circuit, "shots": 1024},
                    reasoning="Circuit ready for simulation"
                )
            return None

        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ANALYZER,
            rules=[simulate_rule],
            tools=["simulate_circuit", "get_statevector", "get_probabilities"]
        )


# Factory function to create all specialized agents
def create_all_agents() -> Dict[str, LLMAgent]:
    """Create instances of all specialized agents."""
    return {
        "architect": ArchitectAgent(),
        "builder": BuilderAgent(),
        "validator": ValidatorAgent(),
        "optimizer": OptimizerAgent(),
        "analyzer": AnalyzerAgent(),
        "scorer": ScorerAgent(),
        "simulator": SimulatorAgent()
    }
