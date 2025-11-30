# Path: QAgents-workflows/ui/mcp_health.py
# Relations: Uses client/mcp_client.py for health checks
#            Used by __init__.py, app.py, chat_components.py
# Description: MCP health monitoring UI components
"""
MCP Health Components: Monitor QuantumArchitect-MCP endpoint availability.

Provides:
- Server health check
- Individual endpoint health checks
- Health status table display
"""

import os
import gradio as gr
import requests
import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Default MCP Server URL
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL",
    "https://mcp-1st-birthday-quantumarchitect-mcp.hf.space"
)

# MCP Endpoints definitions
MCP_ENDPOINTS = [
    {"name": "create_circuit", "category": "Creation", "description": "Create circuit from template"},
    {"name": "parse_qasm", "category": "Creation", "description": "Parse OpenQASM code"},
    {"name": "build_circuit", "category": "Creation", "description": "Build custom circuit from gates"},
    {"name": "validate_circuit", "category": "Validation", "description": "Full circuit validation"},
    {"name": "check_hardware", "category": "Validation", "description": "Hardware compatibility check"},
    {"name": "simulate", "category": "Simulation", "description": "Simulate with measurements"},
    {"name": "get_statevector", "category": "Simulation", "description": "Extract statevector"},
    {"name": "estimate_fidelity", "category": "Simulation", "description": "Hardware fidelity estimation"},
    {"name": "score_circuit", "category": "Scoring", "description": "Circuit scoring metrics"},
    {"name": "compare_circuits", "category": "Scoring", "description": "Compare multiple circuits"},
    {"name": "get_gate_info", "category": "Documentation", "description": "Gate documentation"},
    {"name": "get_algorithm_info", "category": "Documentation", "description": "Algorithm documentation"},
    {"name": "list_hardware", "category": "Documentation", "description": "Available hardware profiles"},
    {"name": "list_templates", "category": "Documentation", "description": "Available circuit templates"},
]


def check_mcp_health(server_url: str = None) -> str:
    """
    Check overall MCP server health.
    
    Args:
        server_url: MCP server URL (uses default if not provided)
        
    Returns:
        Status string with emoji indicator
    """
    url = server_url or MCP_SERVER_URL
    try:
        response = requests.get(f"{url}/", timeout=10)
        if response.status_code == 200:
            return f"🟢 **Connected** to MCP Server"
        else:
            return f"🟡 **Partial** - Status {response.status_code}"
    except requests.exceptions.Timeout:
        return "🟠 **Timeout** - Server slow to respond"
    except requests.exceptions.ConnectionError:
        return "🔴 **Disconnected** - Cannot reach server"
    except Exception as e:
        return f"🔴 **Error**: {str(e)[:50]}"


def check_endpoint_health(endpoint_name: str, server_url: str = None) -> Dict:
    """
    Check health of a specific MCP endpoint.
    
    Args:
        endpoint_name: Name of the endpoint to check
        server_url: MCP server URL (uses default if not provided)
        
    Returns:
        Dict with status, latency_ms, and error fields
    """
    url = server_url or MCP_SERVER_URL
    start = time.perf_counter()
    try:
        endpoint_url = f"{url}/gradio_api/call/ui_{endpoint_name}"
        response = requests.post(endpoint_url, json={"data": []}, timeout=15)
        elapsed = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            return {"status": "🟢", "latency_ms": round(elapsed, 1), "error": None}
        elif response.status_code == 404:
            return {"status": "🟡", "latency_ms": round(elapsed, 1), "error": "Not found"}
        else:
            return {"status": "🟠", "latency_ms": round(elapsed, 1), "error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "🟠", "latency_ms": 15000, "error": "Timeout"}
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"status": "🔴", "latency_ms": round(elapsed, 1), "error": str(e)[:50]}


def get_all_endpoints_health(server_url: str = None) -> str:
    """
    Get health status of all MCP endpoints as formatted markdown.
    
    Args:
        server_url: MCP server URL (uses default if not provided)
        
    Returns:
        Markdown formatted table with endpoint health status
    """
    url = server_url or MCP_SERVER_URL
    output_lines = [
        "## 🔗 MCP Endpoints Health Check",
        f"**Server:** `{url}`\n",
        "| Endpoint | Category | Status | Latency | Error |",
        "|----------|----------|--------|---------|-------|"
    ]

    for endpoint in MCP_ENDPOINTS:
        health = check_endpoint_health(endpoint["name"], url)
        error_str = health["error"] or "-"
        output_lines.append(
            f"| `{endpoint['name']}` | {endpoint['category']} | {health['status']} | {health['latency_ms']}ms | {error_str} |"
        )

    output_lines.append(f"\n**Last checked:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return "\n".join(output_lines)


def create_mcp_health_tab(server_url: str = None) -> gr.Markdown:
    """
    Create the MCP Health monitoring tab components.
    
    Args:
        server_url: MCP server URL (uses default if not provided)
        
    Returns:
        Health display Markdown component
    """
    url = server_url or MCP_SERVER_URL
    
    gr.Markdown("""
## 🔗 MCP Endpoints Health Monitor

Monitor the health and availability of QuantumArchitect-MCP endpoints.
    """)

    with gr.Row():
        check_all_btn = gr.Button("🔄 Check All Endpoints", variant="primary")

    health_display = gr.Markdown(value="Click 'Check All Endpoints' to start health check...")

    gr.Markdown("---")
    gr.Markdown("### 🔍 Check Single Endpoint")

    with gr.Row():
        endpoint_dropdown = gr.Dropdown(
            choices=[ep["name"] for ep in MCP_ENDPOINTS],
            label="Select Endpoint",
            value=None,
            scale=3
        )
        check_single_btn = gr.Button("Check", scale=1)

    single_result = gr.Markdown(value="")

    # Event handlers
    def check_all_handler():
        return get_all_endpoints_health(url)

    def check_single_handler(endpoint_name: str) -> str:
        if not endpoint_name:
            return "Please select an endpoint."
        health = check_endpoint_health(endpoint_name, url)
        return f"**{endpoint_name}**: {health['status']} ({health['latency_ms']}ms) - {health['error'] or 'OK'}"

    check_all_btn.click(check_all_handler, outputs=[health_display])
    check_single_btn.click(check_single_handler, [endpoint_dropdown], [single_result])

    return health_display
