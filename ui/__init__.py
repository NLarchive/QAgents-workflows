# Path: QAgents-workflows/ui/__init__.py
# Relations: Used by app.py, imports from chat_components.py, mcp_health.py, quick_build.py
# Description: UI module initialization - exports all Gradio components for the app
"""
UI Module: Gradio 6.0 compatible UI components for QAgents-Workflows.

This module provides modular UI components that can be assembled
into the main Gradio app. Each component is designed to work
with the agent orchestration system.
"""

from .chat_components import (
    create_chat_tab,
    chat_response,
    generate_circuit_with_naked,
)

from .mcp_health import (
    create_mcp_health_tab,
    check_mcp_health,
    check_endpoint_health,
    get_all_endpoints_health,
    MCP_ENDPOINTS,
)

from .quick_build import (
    create_quick_build_tab,
    quick_build_circuit,
)

from .styles import CUSTOM_CSS

__all__ = [
    # Chat
    "create_chat_tab",
    "chat_response", 
    "generate_circuit_with_naked",
    # MCP Health
    "create_mcp_health_tab",
    "check_mcp_health",
    "check_endpoint_health",
    "get_all_endpoints_health",
    "MCP_ENDPOINTS",
    # Quick Build
    "create_quick_build_tab",
    "quick_build_circuit",
    # Styles
    "CUSTOM_CSS",
]
