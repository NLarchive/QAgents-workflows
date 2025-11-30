# Path: QAgents-workflows/ui/chat_components.py
# Relations: Uses orchestrators/orchestrator.py (NakedOrchestrator)
#            Used by __init__.py, app.py
# Description: Chat UI components for interacting with quantum circuit agents
"""
Chat Components: Gradio 6.0 compatible chat interface for QAgents.

Provides:
- Chat tab creation
- Message handling with NAKED mode orchestrator
- Help and status commands
"""

import gradio as gr
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def generate_circuit_with_naked(prompt: str) -> str:
    """
    Generate a quantum circuit using NAKED mode (direct LLM call).
    This is the simplest and fastest mode.
    
    Args:
        prompt: User's natural language request
        
    Returns:
        Formatted response with circuit or error message
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


def get_help_text() -> str:
    """Return help text for the chat interface."""
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

💡 **Tip:** Be specific about the number of qubits and desired operations!

**Commands:**
- `help` - Show this help message
- `status` - Check system status"""


def get_status_text(mcp_server_url: str) -> str:
    """Return status text for the chat interface."""
    try:
        from ui.mcp_health import check_mcp_health
        health = check_mcp_health()
        return f"## 📊 System Status\n\n{health}\n\n**MCP Server:** `{mcp_server_url}`"
    except Exception as e:
        return f"## 📊 System Status\n\n🔴 **Error checking status**: {str(e)}"


def chat_response(message: str, history: List, mcp_server_url: str = "") -> str:
    """
    Handle chat messages and generate responses.
    Uses NAKED mode for circuit generation.
    
    Args:
        message: User's message
        history: Chat history
        mcp_server_url: URL of the MCP server
        
    Returns:
        Bot's response message
    """
    if not message.strip():
        return ""

    message_lower = message.lower().strip()

    # Help command
    if message_lower in ['help', '/help', '?']:
        return get_help_text()

    # Status command
    if message_lower in ['status', '/status']:
        return get_status_text(mcp_server_url)

    # Generate circuit
    logger.info(f"Generating circuit for: {message}")
    return generate_circuit_with_naked(message)


def create_chat_tab(mcp_server_url: str = "") -> Tuple[gr.Chatbot, gr.Textbox, gr.Button]:
    """
    Create the chat tab components for Gradio 6.0.
    
    Args:
        mcp_server_url: URL of the MCP server for status checks
        
    Returns:
        Tuple of (chatbot, textbox, send_button) components
    """
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

        bot_response = chat_response(message, chat_history, mcp_server_url)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history

    def show_help(chat_history: List):
        help_text = get_help_text()
        chat_history.append({"role": "user", "content": "help"})
        chat_history.append({"role": "assistant", "content": help_text})
        return chat_history

    def show_status(chat_history: List):
        status_text = get_status_text(mcp_server_url)
        chat_history.append({"role": "user", "content": "status"})
        chat_history.append({"role": "assistant", "content": status_text})
        return chat_history

    # Wire up events
    send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
    clear_btn.click(lambda: [], outputs=[chatbot])
    help_btn.click(show_help, [chatbot], [chatbot])
    status_btn.click(show_status, [chatbot], [chatbot])

    return chatbot, msg_input, send_btn
