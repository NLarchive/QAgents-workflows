# Path: QAgents-workflows/ui/quick_build.py
# Relations: Uses client/mcp_client.py for circuit generation
#            Used by __init__.py, app.py
# Description: Quick circuit builder UI components
"""
Quick Build Components: Fast circuit generation from templates.

Provides:
- Template selection dropdown
- Qubit count slider
- Circuit generation via MCP client
"""

import os
import gradio as gr
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default MCP Server URL
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL",
    "https://mcp-1st-birthday-quantumarchitect-mcp.hf.space"
)

# Available templates
CIRCUIT_TEMPLATES = [
    "bell_state",
    "ghz_state", 
    "qft",
    "grover",
    "superposition",
]


def quick_build_circuit(template: str, num_qubits: int, server_url: Optional[str] = None) -> str:
    """
    Generate a circuit from template using MCP client.
    
    Args:
        template: Template name (bell_state, ghz_state, etc.)
        num_qubits: Number of qubits
        server_url: MCP server URL (uses default if not provided)
        
    Returns:
        QASM code or error message
    """
    url = server_url or MCP_SERVER_URL
    try:
        from client.mcp_client import get_client
        mcp_client = get_client(url)
        result = mcp_client.create_circuit_from_template(template, int(num_qubits))

        if result.success and result.data:
            if isinstance(result.data, dict) and 'qasm' in result.data:
                return result.data['qasm']
            return str(result.data)
        return f"# Error: {result.error or 'Unknown error'}"
    except Exception as e:
        logger.error(f"Quick build error: {e}")
        return f"# Error: {str(e)}"


def create_quick_build_tab(server_url: Optional[str] = None) -> gr.Code:
    """
    Create the Quick Build tab components.
    
    Args:
        server_url: MCP server URL (uses default if not provided)
        
    Returns:
        QASM output Code component
    """
    url = server_url or MCP_SERVER_URL
    
    gr.Markdown("""
## 🛠️ Quick Circuit Builder

Generate circuits directly from templates.
    """)

    with gr.Row():
        with gr.Column():
            template_select = gr.Dropdown(
                choices=CIRCUIT_TEMPLATES,
                value="bell_state",
                label="Circuit Template"
            )
            qubits_slider = gr.Slider(
                minimum=2,
                maximum=8,
                value=2,
                step=1,
                label="Number of Qubits"
            )
            build_btn = gr.Button("⚡ Generate Circuit", variant="primary")

        with gr.Column():
            qasm_output = gr.Code(
                label="Generated QASM",
                language="python",
                lines=15
            )

    # Event handler
    def build_handler(template: str, num_qubits: int) -> str:
        return quick_build_circuit(template, num_qubits, url)

    build_btn.click(build_handler, [template_select, qubits_slider], [qasm_output])

    return qasm_output
