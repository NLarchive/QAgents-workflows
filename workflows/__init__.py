"""Workflows module: Predefined workflow definitions."""

from .workflow_definitions import (
    WorkflowStatus,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowExecution,
    # Predefined workflows
    BUILD_WORKFLOW,
    OPTIMIZE_WORKFLOW,
    EVALUATE_WORKFLOW,
    FULL_PIPELINE_WORKFLOW,
    WORKFLOWS,
    get_workflow,
    list_workflows
)

__all__ = [
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowDefinition",
    "WorkflowExecution",
    "BUILD_WORKFLOW",
    "OPTIMIZE_WORKFLOW",
    "EVALUATE_WORKFLOW",
    "FULL_PIPELINE_WORKFLOW",
    "WORKFLOWS",
    "get_workflow",
    "list_workflows"
]
