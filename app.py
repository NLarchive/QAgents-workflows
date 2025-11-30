"""
QAgents-Workflows: Hugging Face Space Entry Point
Path: QAgents-workflows/app.py
Related: client/mcp_client.py (MCP connection), orchestrators/ (agent orchestration)

Provides a Gradio interface with:
- Chat UI for interacting with quantum circuit agents
- MCP Endpoints health monitoring tab
- Circuit generation and validation tools

Reads all configuration from environment variables for HF Space deployment.
"""

import os
import gradio as gr
import logging
import requests
import time
from typing import Optional, List, Tuple
from config import LLMConfig
from orchestrators import create_orchestrator
from client.mcp_client import get_client, MCPClient

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

# Initialize MCP client
mcp_client = get_client(MCP_SERVER_URL)

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
# CHAT FUNCTIONALITY
# =============================================================================

def format_circuit_response(result) -> str:
    """Format circuit generation result for chat display."""
    if result.success:
        output = f"✅ **Circuit Generated Successfully** ({result.execution_time_ms:.0f}ms)\n\n"
        if result.final_output:
            output += f"```qasm\n{result.final_output}\n```\n"
        if hasattr(result, 'metrics') and result.metrics:
            output += f"\n**Metrics:** {result.metrics}"
        return output
    else:
        error_msg = "\n".join(result.errors) if hasattr(result, 'errors') else str(result)
        return f"❌ **Generation Failed**\n\n{error_msg}"


def chat_with_agent(
    message: str, 
    history: List[Tuple[str, str]], 
    mode: str,
    difficulty: str
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process user message and generate response from quantum circuit agent.
    
    Args:
        message: User's input message
        history: Chat history [(user_msg, bot_msg), ...]
        mode: Orchestration mode (naked, quasar, hybrid, blackboard)
        difficulty: Problem difficulty level
    
    Returns:
        Tuple of (bot_response, updated_history)
    """
    try:
        logger.info(f"Chat: mode={mode}, difficulty={difficulty}")
        logger.info(f"User message: {message}")
        
        # Check for special commands
        message_lower = message.lower().strip()
        
        # Help command
        if message_lower in ['help', '/help', '?']:
            response = """## 🤖 QAgents Help

I can help you with quantum circuits! Try asking me to:

**Create Circuits:**
- "Create a Bell state"
- "Generate a 3-qubit GHZ state"
- "Make a QFT circuit for 4 qubits"
- "Build a Grover search circuit"

**Analyze Circuits:**
- Paste QASM code and ask "validate this circuit"
- "Score this circuit for IBM hardware"

**Learn:**
- "Explain the Hadamard gate"
- "How does quantum entanglement work?"

**Settings:**
- Use the dropdowns to change orchestration mode and difficulty
- `naked`: Direct LLM + MCP tools
- `quasar`: Structured workflow with validation
- `hybrid`: Combined approach
- `blackboard`: Multi-agent collaboration

💡 **Tip:** Start with simple requests and increase complexity!
"""
            history.append((message, response))
            return response, history
        
        # Status command
        if message_lower in ['status', '/status']:
            mcp_status = check_mcp_health()
            response = f"## 📊 System Status\n\n{mcp_status}"
            history.append((message, response))
            return response, history
        
        # Generate circuit using orchestrator
        orch = create_orchestrator(mode.lower())
        result = orch.run(message)
        
        if result.success:
            response = f"✅ **Success** ({result.execution_time_ms:.0f}ms)\n\n"
            if result.final_output:
                # Check if output looks like QASM
                if 'OPENQASM' in str(result.final_output) or 'qreg' in str(result.final_output):
                    response += f"```qasm\n{result.final_output}\n```"
                else:
                    response += result.final_output
            else:
                response += "Circuit generated but no QASM output available."
            
            # Add step info if available
            if result.steps_completed > 0:
                response += f"\n\n*Completed {result.steps_completed} steps*"
        else:
            error_msg = "\n".join(result.errors) if result.errors else "Unknown error"
            response = f"❌ **Failed** ({result.execution_time_ms:.0f}ms)\n\n{error_msg}"
        
        history.append((message, response))
        return response, history
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_response = f"❌ **System Error**\n\n{str(e)}"
        history.append((message, error_response))
        return error_response, history


# =============================================================================
# MCP HEALTH CHECK FUNCTIONS
# =============================================================================

def check_mcp_health() -> str:
    """Check overall MCP server health."""
    try:
        response = requests.head(f"{MCP_SERVER_URL}/", timeout=10)
        if response.status_code == 200:
            return f"🟢 **Connected** to `{MCP_SERVER_URL}`"
        else:
            return f"🟡 **Partial** - Status {response.status_code}"
    except requests.exceptions.Timeout:
        return f"🟠 **Timeout** - Server slow to respond"
    except requests.exceptions.ConnectionError:
        return f"🔴 **Disconnected** - Cannot reach server"
    except Exception as e:
        return f"🔴 **Error**: {str(e)}"


def check_endpoint_health(endpoint_name: str) -> dict:
    """Check health of a specific MCP endpoint."""
    start = time.perf_counter()
    try:
        # Try to call the Gradio API endpoint
        url = f"{MCP_SERVER_URL}/gradio_api/call/ui_{endpoint_name}"
        response = requests.post(
            url, 
            json={"data": []}, 
            timeout=15
        )
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
    
    # Check server first
    server_status = check_mcp_health()
    
    for endpoint in MCP_ENDPOINTS:
        health = check_endpoint_health(endpoint["name"])
        error_str = health["error"] or "-"
        output_lines.append(
            f"| `{endpoint['name']}` | {endpoint['category']} | {health['status']} | {health['latency_ms']}ms | {error_str} |"
        )
    
    output_lines.append(f"\n**Last checked:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    output_lines.append(f"\n**Server Status:** {server_status}")
    
    return "\n".join(output_lines)


def refresh_health_check():
    """Refresh the health check display."""
    return get_all_endpoints_health()


def check_single_endpoint(endpoint_name: str) -> str:
    """Check a single endpoint and return formatted result."""
    if not endpoint_name:
        return "Please select an endpoint to check."
    
    health = check_endpoint_health(endpoint_name)
    
    output = f"## Endpoint: `{endpoint_name}`\n\n"
    output += f"- **Status:** {health['status']}\n"
    output += f"- **Latency:** {health['latency_ms']}ms\n"
    if health['error']:
        output += f"- **Error:** {health['error']}\n"
    output += f"\n*Checked at {time.strftime('%H:%M:%S')}*"
    
    return output


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Custom CSS for better chat appearance
CUSTOM_CSS = """
.chat-container {
    max-height: 600px;
    overflow-y: auto;
}
.status-panel {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 10px;
    padding: 15px;
}
"""

with gr.Blocks(
    title="QAgents - Quantum Circuit Assistant"
) as demo:
    
    # Header
    gr.Markdown("""
    # ⚛️ QAgents: Quantum Circuit Assistant
    
    Multi-agent system for generating optimized quantum circuits using MCP tools.
    
    **Connected to:** [QuantumArchitect-MCP](https://huggingface.co/spaces/MCP-1st-Birthday/QuantumArchitect-MCP) on HuggingFace
    """)
    
    with gr.Tabs():
        # =================================================================
        # TAB 1: CHAT INTERFACE
        # =================================================================
        with gr.TabItem("💬 Chat", id="chat-tab"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Quantum Circuit Agent",
                        height=500,
                        show_label=True,
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Ask me to create a quantum circuit... (e.g., 'Create a Bell state')",
                            lines=2,
                            scale=4,
                        )
                        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        help_btn = gr.Button("❓ Help", size="sm")
                        status_btn = gr.Button("📊 Status", size="sm")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    
                    mode_select = gr.Dropdown(
                        choices=["naked", "quasar", "hybrid", "blackboard"],
                        value="naked",
                        label="Orchestration Mode",
                        info="How agents collaborate"
                    )
                    
                    difficulty_select = gr.Dropdown(
                        choices=["EASY", "MEDIUM", "HARD", "VERY_HARD"],
                        value="EASY",
                        label="Complexity Level",
                        info="Expected circuit complexity"
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 📡 Quick Status")
                    mcp_status_display = gr.Markdown(value="⏳ Checking...")
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            # Chat event handlers
            def user_submit(message, history, mode, difficulty):
                if not message.strip():
                    return "", history
                response, new_history = chat_with_agent(message, history, mode, difficulty)
                return "", new_history
            
            send_btn.click(
                fn=user_submit,
                inputs=[msg_input, chatbot, mode_select, difficulty_select],
                outputs=[msg_input, chatbot]
            )
            
            msg_input.submit(
                fn=user_submit,
                inputs=[msg_input, chatbot, mode_select, difficulty_select],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot]
            )
            
            help_btn.click(
                fn=lambda h: chat_with_agent("help", h, "naked", "EASY"),
                inputs=[chatbot],
                outputs=[msg_input, chatbot]
            )
            
            status_btn.click(
                fn=lambda h: chat_with_agent("status", h, "naked", "EASY"),
                inputs=[chatbot],
                outputs=[msg_input, chatbot]
            )
            
            refresh_btn.click(
                fn=check_mcp_health,
                outputs=[mcp_status_display]
            )
        
        # =================================================================
        # TAB 2: MCP ENDPOINTS HEALTH
        # =================================================================
        with gr.TabItem("🔗 MCP Endpoints", id="mcp-tab"):
            gr.Markdown("""
            ## 🔗 MCP Endpoints Health Monitor
            
            Monitor the health and availability of QuantumArchitect-MCP endpoints.
            
            **MCP Server:** [QuantumArchitect-MCP](https://huggingface.co/spaces/MCP-1st-Birthday/QuantumArchitect-MCP)
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
            
            gr.Markdown("---")
            gr.Markdown("""
            ### 📚 Available MCP Endpoints
            
            | Category | Endpoints |
            |----------|-----------|
            | **Creation** | `create_circuit`, `parse_qasm`, `build_circuit` |
            | **Validation** | `validate_circuit`, `check_hardware` |
            | **Simulation** | `simulate`, `get_statevector`, `estimate_fidelity` |
            | **Scoring** | `score_circuit`, `compare_circuits` |
            | **Documentation** | `get_gate_info`, `get_algorithm_info`, `list_hardware`, `list_templates` |
            
            All endpoints accept JSON and return JSON via the Gradio MCP protocol.
            """)
            
            # Event handlers
            check_all_btn.click(
                fn=refresh_health_check,
                outputs=[health_display]
            )
            
            check_single_btn.click(
                fn=check_single_endpoint,
                inputs=[endpoint_dropdown],
                outputs=[single_result]
            )
        
        # =================================================================
        # TAB 3: QUICK CIRCUIT BUILDER
        # =================================================================
        with gr.TabItem("🛠️ Quick Build", id="build-tab"):
            gr.Markdown("""
            ## 🛠️ Quick Circuit Builder
            
            Generate circuits directly from templates without chat.
            """)
            
            with gr.Row():
                with gr.Column():
                    template_select = gr.Dropdown(
                        choices=["bell_state", "ghz_state", "qft", "grover", "vqe", "superposition"],
                        value="bell_state",
                        label="Circuit Template"
                    )
                    qubits_slider = gr.Slider(
                        minimum=2,
                        maximum=10,
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
            
            def quick_build(template, num_qubits):
                try:
                    result = mcp_client.create_circuit_from_template(template, int(num_qubits))
                    if result.success and result.data:
                        if isinstance(result.data, dict) and 'qasm' in result.data:
                            return result.data['qasm']
                        return str(result.data)
                    return f"# Error: {result.error or 'Unknown error'}"
                except Exception as e:
                    return f"# Error: {str(e)}"
            
            build_btn.click(
                fn=quick_build,
                inputs=[template_select, qubits_slider],
                outputs=[qasm_output]
            )
        
        # =================================================================
        # TAB 4: ABOUT
        # =================================================================
        with gr.TabItem("ℹ️ About", id="about-tab"):
            gr.Markdown("""
            ## ℹ️ About QAgents
            
            **QAgents** is a multi-agent system for quantum circuit generation and optimization.
            
            ### 🏗️ Architecture
            
            ```
            ┌─────────────────────────────────────────────┐
            │           QAgents-Workflows                 │
            │  (This App - Agent Orchestration)           │
            ├─────────────────────────────────────────────┤
            │  • Chat Interface                           │
            │  • Multi-mode Orchestrators                 │
            │  • LLM Integration (Gemini/LiteLLM)         │
            └─────────────────┬───────────────────────────┘
                              │ MCP Protocol
                              ▼
            ┌─────────────────────────────────────────────┐
            │       QuantumArchitect-MCP                  │
            │  (HuggingFace Space - Circuit Tools)        │
            ├─────────────────────────────────────────────┤
            │  • Circuit Creation & Templates             │
            │  • Validation & Syntax Checking             │
            │  • Simulation & Statevectors                │
            │  • Scoring & Hardware Fitness               │
            └─────────────────────────────────────────────┘
            ```
            
            ### 🤖 Orchestration Modes
            
            | Mode | Description | Best For |
            |------|-------------|----------|
            | **naked** | Direct LLM + MCP tools | Simple queries |
            | **quasar** | Structured workflow | Complex circuits |
            | **hybrid** | Adaptive approach | General use |
            | **blackboard** | Multi-agent collab | Hard problems |
            
            ### 🔗 Links
            
            - [QuantumArchitect-MCP](https://huggingface.co/spaces/MCP-1st-Birthday/QuantumArchitect-MCP)
            - [QAgents Source](https://github.com/NLarchive/QAgents-workflows)
            
            ### 📝 License
            
            MIT License - Feel free to use and modify!
            """)
    
    # Load initial status on app start
    demo.load(
        fn=check_mcp_health,
        outputs=[mcp_status_display]
    )

# Launch configuration for HuggingFace Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        show_error=True,
        mcp_server=True,
        css=CUSTOM_CSS,
    )
