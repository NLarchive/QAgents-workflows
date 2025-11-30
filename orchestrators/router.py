# Path: QAgents-workflos/orchestrators/router.py
# Relations: Used by orchestrators/orchestrator.py, run_quality_eval.py
# Description: Difficulty-aware orchestrator selection based on problem complexity
#              Routes easy problems to NAKED (fastest, best quality)
#              Routes medium to NAKED+optimization, hard to GUIDED

"""
Difficulty-Aware Router: Selects optimal orchestration mode based on problem complexity.

Based on quality evaluation findings:
- NAKED mode: Best for easy problems (47.9/100 quality, 3.7s)
- NAKED+Optimizer: Best for medium (post-generation refinement)
- GUIDED: For hard problems (agents may add value for complex algorithms)

This router balances quality, cost, and execution time.
"""

from typing import Optional, Dict, Literal
from dataclasses import dataclass
from tests.test_problems import TestProblem, ProblemDifficulty


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    mode: Literal["naked", "guided", "blackboard"]
    reason: str
    expected_quality: float
    expected_llm_calls: int
    expected_time_ms: int
    use_optimizer: bool = False


class DifficultyAwareRouter:
    """
    Routes problems to optimal orchestrators based on difficulty and characteristics.
    
    Strategy:
    - EASY: Use NAKED (proven best)
    - MEDIUM: Use NAKED + post-optimization
    - HARD: Use GUIDED if agents help, NAKED+optimizer as fallback
    
    Can be configured for experimentation.
    """
    
    # Routing configuration (can be tuned)
    ROUTING_CONFIG = {
        "easy": {
            "primary_mode": "naked",
            "use_optimizer": False,
            "fallback_mode": "guided",
            "expected_quality": 47.9,
            "expected_llm_calls": 3,
            "expected_time_ms": 3700,
        },
        "medium": {
            "primary_mode": "naked",
            "use_optimizer": True,  # Add post-generation optimization
            "fallback_mode": "guided",
            "expected_quality": 50.0,  # Estimated with optimizer
            "expected_llm_calls": 3,
            "expected_time_ms": 5000,
        },
        "hard": {
            "primary_mode": "guided",  # Agents might help for complex algorithms
            "use_optimizer": True,
            "fallback_mode": "naked",
            "expected_quality": 55.0,  # Estimated
            "expected_llm_calls": 7,
            "expected_time_ms": 25000,
        }
    }
    
    @classmethod
    def route(cls, problem: TestProblem, 
              prefer_naked: bool = False,
              prefer_guided: bool = False) -> RoutingDecision:
        """
        Route a problem to the optimal orchestrator.
        
        Args:
            problem: The quantum circuit problem to solve
            prefer_naked: Force NAKED mode (for testing)
            prefer_guided: Force GUIDED mode (for testing)
            
        Returns:
            RoutingDecision with selected mode and metadata
        """
        
        # Handle overrides
        if prefer_naked:
            return cls._make_decision("naked", problem, "User override")
        if prefer_guided:
            return cls._make_decision("guided", problem, "User override")
        
        # Get difficulty level
        difficulty = problem.difficulty.value if hasattr(problem.difficulty, 'value') else str(problem.difficulty)
        
        # Get routing config for difficulty
        config = cls.ROUTING_CONFIG.get(difficulty)
        if not config:
            # Default to guided for unknown difficulties
            return cls._make_decision("guided", problem, f"Unknown difficulty: {difficulty}")
        
        # Route based on difficulty
        return cls._make_decision(
            config["primary_mode"],
            problem,
            f"Routed based on difficulty: {difficulty}",
            use_optimizer=config.get("use_optimizer", False),
            expected_quality=config["expected_quality"],
            expected_llm_calls=config["expected_llm_calls"],
            expected_time_ms=config["expected_time_ms"],
        )
    
    @classmethod
    def route_batch(cls, problems: list) -> Dict[str, RoutingDecision]:
        """Route multiple problems."""
        return {p.id: cls.route(p) for p in problems}
    
    @classmethod
    def _make_decision(cls, mode: str, problem: TestProblem, reason: str,
                      use_optimizer: bool = False,
                      expected_quality: float = 45.0,
                      expected_llm_calls: int = 3,
                      expected_time_ms: int = 5000) -> RoutingDecision:
        """Create a routing decision."""
        return RoutingDecision(
            mode=mode,
            reason=reason,
            expected_quality=expected_quality,
            expected_llm_calls=expected_llm_calls,
            expected_time_ms=expected_time_ms,
            use_optimizer=use_optimizer,
        )
    
    @classmethod
    def print_strategy(cls):
        """Print routing strategy."""
        print("\n" + "="*80)
        print("DIFFICULTY-AWARE ROUTING STRATEGY")
        print("="*80)
        
        for difficulty in ["easy", "medium", "hard"]:
            config = cls.ROUTING_CONFIG[difficulty]
            print(f"\n{difficulty.upper()}:")
            print(f"  Primary Mode: {config['primary_mode']}")
            print(f"  Use Optimizer: {config['use_optimizer']}")
            print(f"  Fallback: {config['fallback_mode']}")
            print(f"  Expected Quality: {config['expected_quality']:.1f}/100")
            print(f"  Expected LLM Calls: {config['expected_llm_calls']}")
            print(f"  Expected Time: {config['expected_time_ms']}ms")
        
        print("\n" + "="*80)


def select_orchestrator_mode(problem: TestProblem) -> str:
    """
    Convenience function: Get orchestrator mode for a problem.
    
    Usage:
        mode = select_orchestrator_mode(problem)
        orchestrator = create_orchestrator(mode)
    """
    decision = DifficultyAwareRouter.route(problem)
    return decision.mode


def should_use_optimizer(problem: TestProblem) -> bool:
    """Check if optimization should be applied after generation."""
    decision = DifficultyAwareRouter.route(problem)
    return decision.use_optimizer


# Example usage
if __name__ == "__main__":
    from tests.test_problems import EASY_PROBLEMS, MEDIUM_PROBLEMS, HARD_PROBLEMS
    
    print("\nExample: Routing all problems")
    print("-" * 80)
    
    all_problems = EASY_PROBLEMS + MEDIUM_PROBLEMS + HARD_PROBLEMS
    
    for problem in all_problems:
        decision = DifficultyAwareRouter.route(problem)
        print(f"{problem.id:15} -> {decision.mode:10} ({decision.reason})")
    
    DifficultyAwareRouter.print_strategy()
