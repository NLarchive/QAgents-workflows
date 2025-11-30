"""Prompts module: System prompts for all agents."""

from .agent_prompts import (
    ARCHITECT_PROMPT,
    BUILDER_PROMPT,
    VALIDATOR_PROMPT,
    OPTIMIZER_PROMPT,
    ANALYZER_PROMPT,
    SCORER_PROMPT,
    COORDINATOR_PROMPT,
    ALL_PROMPTS,
    get_prompt
)

__all__ = [
    "ARCHITECT_PROMPT",
    "BUILDER_PROMPT", 
    "VALIDATOR_PROMPT",
    "OPTIMIZER_PROMPT",
    "ANALYZER_PROMPT",
    "SCORER_PROMPT",
    "COORDINATOR_PROMPT",
    "ALL_PROMPTS",
    "get_prompt"
]
