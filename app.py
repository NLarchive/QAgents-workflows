"""
QAgents-Workflows: Hugging Face Space Entry Point
Path: QAgents-workflows/app.py
Related: client/mcp_client.py (MCP connection), orchestrators/ (agent orchestration)

Provides a Gradio interface with:
- Chat UI for interacting with quantum circuit agents (NAKED mode)
- MCP Endpoints health monitoring tab
- Circuit generation and validation tools

Reads all configuration from environment variables for HF Space deployment.
"""

import os
import gradio as gr
import logging
import requests
import time
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server URL for QuantumArchitect-MCP on HuggingFace
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL",
    "https://mcp-1st-birthday-quantumarchitect-mcp.hf.space"
)

# Log environment configuration at startup
logger.info("=" * 70)
logger.info("QAgents Quantum Circuit Orchestrator - Initialization")
logger.info("=" * 70)
logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini (default)')}")
logger.info(f"LLM Model: {os.getenv('LLM_MODEL', 'gemini-2.5-flash-lite (default)')}")
logger.info(f"MCP Server URL: {MCP_SERVER_URL}")
logger.info(f"Google API Key configured: {bool(os.getenv('GOOGLE_API_KEY') or os.getenv('GENAI_API_KEY'))}")
logger.info("=" * 70)

# =============================================================================
# MCP ENDPOINTS DEFINITIONS
# =============================================================================

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

# =============================================================================
# MCP HEALTH CHECK FUNCTIONS
# =============================================================================

def check_mcp_health() -> str:
    """Check overall MCP server health."""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/", timeout=10)
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


def check_endpoint_health(endpoint_name: str) -> Dict:
    """Check health of a specific MCP endpoint."""
    start = time.perf_counter()
    try:
        url = f"{MCP_SERVER_URL}/gradio_api/call/ui_{endpoint_name}"
        response = requests.post(url, json={"data": []}, timeout=15)
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


def get_all_endpoints_health() -> str:
    """Get health status of all MCP endpoints as formatted markdown."""
    output_lines = [
        "## 🔗 MCP Endpoints Health Check",
        f"**Server:** `{MCP_SERVER_URL}`\n",
        "| Endpoint | Category | Status | Latency | Error |",
        "|----------|----------|--------|---------|-------|"
    ]

    for endpoint in MCP_ENDPOINTS:
        health = check_endpoint_health(endpoint["name"])
        error_str = health["error"] or "-"
        output_lines.append(
            f"| `{endpoint['name']}` | {endpoint['category']} | {health['status']} | {health['latency_ms']}ms | {error_str} |"
        )

    output_lines.append(f"\n**Last checked:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return "\n".join(output_lines)


# =============================================================================
# CHAT FUNCTIONALITY - Using NAKED mode (direct LLM)
# =============================================================================

def generate_circuit_with_naked(prompt: str) -> str:
    """
    Generate a quantum circuit using NAKED mode (direct LLM call).
    This is the simplest and fastest mode.
    """
    try:
        # Lazy import to avoid startup issues
        from orchestrators import create_orchestrator
        
        orch = create_orchestrator("naked")
        result = orch.run(prompt)
        
        if result.success:
            output = f"✅ **Success** ({result.execution_time_ms:.0f}ms)\n\n"
            if result.final_output:
                if 'OPENQASM' in str(result.final_output) or 'qreg' in str(result.final_output):
                    output += f"```qasm\n{result.final_output}\n```"
                else:
                    output += str(result.final_output)
            return output
        else:
            error_msg = "\n".join(result.errors) if result.errors else "Unknown error"
            return f"❌ **Failed** ({result.execution_time_ms:.0f}ms)\n\n{error_msg}"
    except Exception as e:
        logger.error(f"NAKED mode error: {e}")
        return f"❌ **Error**: {str(e)}"


def chat_response(message: str, history: List) -> str:
    """
    Handle chat messages and generate responses.
    Uses NAKED mode for circuit generation.
    """
    if not message.strip():
        return ""
    
    message_lower = message.lower().strip()
    
    # Help command
    if message_lower in ['help', '/help', '?']:
        return """## 🤖 QAgents Help

I can help you with quantum circuits! Try asking me to:

**Create Circuits:**
- "Create a Bell state"
- "Generate a 3-qubit GHZ state"
- "Make a QFT circuit for 4 qubits"
- "Build a simple superposition"

**Examples:**
- "Create a circuit that puts 2 qubits in superposition"
- "Generate a CNOT gate between qubit 0 and 1"
- "Build a quantum teleportation circuit"

💡 **Tip:** Be specific about the number of qubits and desired operations!"""

    # Status command
    if message_lower in ['status', '/status']:
        return f"## 📊 System Status\n\n{check_mcp_health()}\n\n**MCP Server:** `{MCP_SERVER_URL}`"

    # Generate circuit
    logger.info(f"Generating circuit for: {message}")
    return generate_circuit_with_naked(message)


# =============================================================================
# QUICK BUILD FUNCTIONALITY
# =============================================================================

def quick_build_circuit(template: str, num_qubits: int) -> str:
    """Generate a circuit from template using MCP client."""
    try:
        from client.mcp_client import get_client
        mcp_client = get_client(MCP_SERVER_URL)
        result = mcp_client.create_circuit_from_template(template, int(num_qubits))
        
        if result.success and result.data:
            if isinstance(result.data, dict) and 'qasm' in result.data:
                return result.data['qasm']
            return str(result.data)
        return f"# Error: {result.error or 'Unknown error'}"
    except Exception as e:
        return f"# Error: {str(e)}"


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="QAgents - Quantum Circuit Assistant") as demo:
    
    # Header
    gr.Markdown("""
    # ⚛️ QAgents: Quantum Circuit Assistant
    
    Generate quantum circuits using natural language. Powered by LLM + MCP tools.
    """)
    
    with gr.Tabs():
        # =================================================================
        # TAB 1: CHAT INTERFACE
        # =================================================================
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Chat with Quantum Circuit Agent")
            gr.Markdown("Ask me to create quantum circuits! Try: *'Create a Bell state'* or *'Generate a 3-qubit GHZ state'*")
            
            chatbot = gr.Chatbot(
                value=[],
                height=400,
                label="Quantum Circuit Agent"
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me to create a quantum circuit...",
                    label="Your Message",
                    scale=4,
                    lines=1
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", size="sm")
                help_btn = gr.Button("❓ Help", size="sm")
                status_btn = gr.Button("📊 Status", size="sm")
            
            # Chat handlers
            def respond(message: str, chat_history: List):
                if not message.strip():
                    return "", chat_history
                
                bot_response = chat_response(message, chat_history)
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_response})
                return "", chat_history
            
            def show_help(chat_history: List):
                help_text = chat_response("help", chat_history)
                chat_history.append({"role": "user", "content": "help"})
                chat_history.append({"role": "assistant", "content": help_text})
                return chat_history
            
            def show_status(chat_history: List):
                status_text = chat_response("status", chat_history)
                chat_history.append({"role": "user", "content": "status"})
                chat_history.append({"role": "assistant", "content": status_text})
                return chat_history
            
            send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
            msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
            clear_btn.click(lambda: [], outputs=[chatbot])
            help_btn.click(show_help, [chatbot], [chatbot])
            status_btn.click(show_status, [chatbot], [chatbot])
        
        # =================================================================
        # TAB 2: MCP ENDPOINTS HEALTH
        # =================================================================
        with gr.TabItem("🔗 MCP Health"):
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
            
            def check_single(endpoint_name: str) -> str:
                if not endpoint_name:
                    return "Please select an endpoint."
                health = check_endpoint_health(endpoint_name)
                return f"**{endpoint_name}**: {health['status']} ({health['latency_ms']}ms) - {health['error'] or 'OK'}"
            
            check_all_btn.click(get_all_endpoints_health, outputs=[health_display])
            check_single_btn.click(check_single, [endpoint_dropdown], [single_result])
        
        # =================================================================
        # TAB 3: QUICK BUILD
        # =================================================================
        with gr.TabItem("🛠️ Quick Build"):
            gr.Markdown("""
            ## 🛠️ Quick Circuit Builder
            
            Generate circuits directly from templates.
            """)
            
            with gr.Row():
                with gr.Column():
                    template_select = gr.Dropdown(
                        choices=["bell_state", "ghz_state", "qft", "grover", "superposition"],
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
            
            build_btn.click(quick_build_circuit, [template_select, qubits_slider], [qasm_output])
        
        # =================================================================
        # TAB 4: ABOUT
        # =================================================================
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## ℹ️ About QAgents
            
            **QAgents** is a multi-agent system for quantum circuit generation.
            
            ### 🏗️ How it Works
            
            1. **You ask** for a quantum circuit in natural language
            2. **NAKED mode** uses an LLM to generate OpenQASM code directly
            3. **MCP tools** validate and simulate your circuits
            
            ### 🔗 Architecture
            
            - **Frontend**: This Gradio app (QAgents-Workflows)
            - **Backend**: QuantumArchitect-MCP on HuggingFace
            - **LLM**: Gemini 2.5 Flash (configurable)
            
            ### 📝 License
            
            MIT License - Feel free to use and modify!
            """)

# Launch for HuggingFace Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        show_error=True,
        mcp_server=True
    )
